from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ViewFrameTimeline:
    role: str
    video_path: str
    frames_csv_path: str | None
    duration_sec: float
    fps: float
    absolute_times_sec: tuple[float, ...]
    local_video_secs: tuple[float, ...]
    source: str
    frame_indices: tuple[int, ...] = ()
    timestamp_field: str | None = None
    duplicate_timestamp_count: int = 0
    non_monotonic_timestamp_count: int = 0
    invalid_frame_count: int = 0

    @property
    def start_abs_sec(self) -> float:
        return float(self.absolute_times_sec[0]) if self.absolute_times_sec else 0.0

    @property
    def end_abs_sec(self) -> float:
        return float(self.absolute_times_sec[-1]) if self.absolute_times_sec else float(self.duration_sec)

    def nearest(self, absolute_time_sec: float) -> tuple[int, float, float]:
        if not self.absolute_times_sec:
            local_sec = min(max(0.0, absolute_time_sec - self.start_abs_sec), self.duration_sec)
            return int(round(local_sec * self.fps)), local_sec, 0.0
        idx = bisect.bisect_left(self.absolute_times_sec, float(absolute_time_sec))
        if idx <= 0:
            nearest = 0
        elif idx >= len(self.absolute_times_sec):
            nearest = len(self.absolute_times_sec) - 1
        else:
            before = self.absolute_times_sec[idx - 1]
            after = self.absolute_times_sec[idx]
            nearest = idx if abs(after - absolute_time_sec) < abs(before - absolute_time_sec) else idx - 1
        frame_index = int(self.frame_indices[nearest]) if self.frame_indices else nearest
        return frame_index, float(self.local_video_secs[nearest]), abs(float(self.absolute_times_sec[nearest]) - float(absolute_time_sec))


@dataclass(frozen=True)
class AlignmentSample:
    sample_index: int
    common_time_sec: float
    absolute_time_sec: float
    third_frame_index: int
    first_frame_index: int
    third_local_sec: float
    first_local_sec: float
    third_delta_sec: float
    first_delta_sec: float
    pair_delta_sec: float


def build_aligned_dual_view_videos(
    *,
    third_video: str | Path,
    first_video: str | Path,
    output_dir: str | Path,
    third_frames_csv: str | Path | None = None,
    first_frames_csv: str | Path | None = None,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    target_fps: float | None = None,
    max_pair_delta_sec: float = 0.150,
    max_width: int = 960,
    segment_name: str = "aligned_dual_view",
    max_workers: int = 2,
) -> dict[str, Any]:
    """Build a nearest-neighbor aligned dual-view video pair.

    The output is a preprocessing artifact, not a semantic proof. Formal
    publication still requires the downstream dual-view action-phase gate.
    """

    started = time.perf_counter()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    third = load_view_timeline("third_person", third_video, third_frames_csv)
    first = load_view_timeline("first_person", first_video, first_frames_csv)
    common_start = max(third.start_abs_sec, first.start_abs_sec)
    common_end = min(third.end_abs_sec, first.end_abs_sec)
    requested_start = common_start + max(0.0, float(start_sec))
    requested_end = common_end if duration_sec is None else min(common_end, requested_start + max(0.0, float(duration_sec)))
    if requested_end <= requested_start:
        raise ValueError("No common dual-view time range is available for alignment.")
    fps = float(target_fps or max(1.0, min(third.fps, first.fps)))
    _, reference_times = reference_times_for_common_range(
        third=third,
        first=first,
        start_abs_sec=requested_start,
        end_abs_sec=requested_end,
        target_fps=fps,
    )
    samples = build_alignment_samples(
        third=third,
        first=first,
        unit_id=segment_name,
        start_abs_sec=requested_start,
        end_abs_sec=requested_end,
        target_fps=fps,
        max_pair_delta_sec=max_pair_delta_sec,
    )
    if not samples:
        raise ValueError("No dual-view frame pairs passed the nearest-neighbor synchronization threshold.")
    frame_root = out_dir / "_aligned_frames" / segment_name
    third_frame_dir = frame_root / "third_person"
    first_frame_dir = frame_root / "first_person"
    for directory in (third_frame_dir, first_frame_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("*.jpg"):
            old.unlink()

    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 2))) as pool:
        fut_third = pool.submit(_extract_view_frames_by_index, third.video_path, [sample.third_frame_index for sample in samples], third_frame_dir, max_width)
        fut_first = pool.submit(_extract_view_frames_by_index, first.video_path, [sample.first_frame_index for sample in samples], first_frame_dir, max_width)
        third_extract = fut_third.result()
        first_extract = fut_first.result()

    third_out = out_dir / f"{segment_name}_third_aligned.mp4"
    first_out = out_dir / f"{segment_name}_first_aligned.mp4"
    side_out = out_dir / f"{segment_name}_side_by_side.mp4"
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_third_video = pool.submit(_encode_frames, third_frame_dir, third_out, fps)
        fut_first_video = pool.submit(_encode_frames, first_frame_dir, first_out, fps)
        fut_third_video.result()
        fut_first_video.result()
    _encode_side_by_side(third_out, first_out, side_out)

    quality = alignment_quality_report(
        third=third,
        first=first,
        samples=samples,
        target_fps=fps,
        source_sample_count=len(reference_times),
        max_pair_delta_sec=max_pair_delta_sec,
        output_paths={
            "aligned_third_video": str(third_out),
            "aligned_first_video": str(first_out),
            "aligned_side_by_side_video": str(side_out),
        },
    )
    quality["extract_timing_sec"] = {
        "third_person": round(float(third_extract.get("elapsed_sec", 0.0)), 6),
        "first_person": round(float(first_extract.get("elapsed_sec", 0.0)), 6),
        "total_wall_sec": round(time.perf_counter() - started, 6),
    }
    (out_dir / f"{segment_name}_alignment_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_alignment_manifest(out_dir / f"{segment_name}_alignment_manifest.json", third, first, samples, quality)
    write_alignment_index(out_dir / f"{segment_name}_alignment_index.csv", samples, third=third, first=first, unit_id=segment_name)
    write_alignment_index(out_dir / f"{segment_name}_alignment_index.jsonl", samples, third=third, first=first, unit_id=segment_name)
    return quality


def analyze_dual_view_frame_alignment(
    *,
    third_video: str | Path,
    first_video: str | Path,
    third_frames_csv: str | Path | None = None,
    first_frames_csv: str | Path | None = None,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    target_fps: float | None = None,
    max_pair_delta_sec: float = 0.150,
) -> dict[str, Any]:
    """Assess nearest-neighbor frame alignment without extracting pixels.

    This is the cheap pre-YOLO pipeline gate. It validates whether a common
    session timeline can be mapped into legal local PTS positions for both
    views. It does not claim semantic/action alignment by itself.
    """

    started = time.perf_counter()
    third = load_view_timeline("third_person", third_video, third_frames_csv)
    first = load_view_timeline("first_person", first_video, first_frames_csv)
    common_start = max(third.start_abs_sec, first.start_abs_sec)
    common_end = min(third.end_abs_sec, first.end_abs_sec)
    requested_start = common_start + max(0.0, float(start_sec))
    requested_end = common_end if duration_sec is None else min(common_end, requested_start + max(0.0, float(duration_sec)))
    if requested_end <= requested_start:
        return {
            "schema_version": "dual_view_nearest_neighbor_alignment.v1",
            "status": "frame_time_alignment_unreliable",
            "formal_results_allowed": False,
            "video_memory_allowed": False,
            "reasons": ["no_common_dual_view_frame_time_range"],
            "target_fps": float(target_fps or 0.0),
            "sample_count": 0,
            "third_person": _timeline_summary(third),
            "first_person": _timeline_summary(first),
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    fps = float(target_fps or max(1.0, min(third.fps, first.fps)))
    _, reference_times = reference_times_for_common_range(
        third=third,
        first=first,
        start_abs_sec=requested_start,
        end_abs_sec=requested_end,
        target_fps=fps,
    )
    samples = build_alignment_samples(
        third=third,
        first=first,
        unit_id="preflight",
        start_abs_sec=requested_start,
        end_abs_sec=requested_end,
        target_fps=fps,
        max_pair_delta_sec=max_pair_delta_sec,
    )
    payload = alignment_quality_report(
        third=third,
        first=first,
        samples=samples,
        target_fps=fps,
        source_sample_count=len(reference_times),
        max_pair_delta_sec=max_pair_delta_sec,
        output_paths={},
    )
    status = str(payload.get("status") or "")
    payload["formal_results_allowed"] = status != "frame_time_alignment_unreliable"
    payload["video_memory_allowed"] = status != "frame_time_alignment_unreliable"
    payload["reasons"] = [] if payload["formal_results_allowed"] else ["nearest_neighbor_frame_time_alignment_unreliable"]
    payload["elapsed_sec"] = round(time.perf_counter() - started, 6)
    return payload


def run_dual_view_alignment_pipeline(
    manifest: Any,
    output_dir: str | Path,
    *,
    timestamp_field: str | None = None,
    max_delta_ms: float = 300.0,
    median_gate_ms: float = 50.0,
    p90_gate_ms: float = 150.0,
    make_aligned_videos: bool = False,
    target_fps: float | None = None,
) -> dict[str, Any]:
    """Run the dual-view timestamp calibration artifacts before YOLO analysis.

    This function is intentionally an adapter around existing pipeline data:
    it reads ``SessionManifest`` style video sources and writes durable
    artifacts under the current run output directory. It does not move,
    rewrite, or delete raw inputs.
    """

    started = time.perf_counter()
    root = Path(output_dir) / "dual_view_alignment"
    root.mkdir(parents=True, exist_ok=True)
    sources = manifest.videos.all_sources()
    third_source = sources.get("third_person")
    first_source = sources.get("first_person")
    summary: dict[str, Any] = {
        "schema_version": "dual_view_alignment_pipeline.v1",
        "stage_order": [
            "video_registration",
            "single_view_time_axis_health",
            "dual_view_state_scan",
            "trim_irrelevant_frames",
            "alignment_units",
            "local_offset_estimation",
            "nearest_neighbor_sync_index",
            "aligned_video_rebuild",
            "alignment_quality_report",
            "action_phase_consistency",
            "formal_experiment_windows",
            "fine_scan_ready",
            "physical_action_extraction_ready",
            "key_material_generation_ready",
            "publish_gate_ready",
            "material_library_and_video_memory_ready",
        ],
        "formal_results_allowed": False,
        "video_memory_allowed": False,
        "artifacts": {},
        "warnings": [],
    }
    if third_source is None or first_source is None:
        summary.update(
            {
                "status": "missing_dual_view_source",
                "warnings": ["first_person_or_third_person_source_missing"],
            }
        )
        _write_json(root / "dual_view_alignment_pipeline_summary.json", summary)
        return summary

    third = load_view_timeline(
        "third_person",
        getattr(third_source, "path"),
        _source_frames_csv_path(third_source),
        timestamp_field=timestamp_field,
    )
    first = load_view_timeline(
        "first_person",
        getattr(first_source, "path"),
        _source_frames_csv_path(first_source),
        timestamp_field=timestamp_field,
    )

    registration_rows = [
        _video_registration_row(manifest, third_source, third),
        _video_registration_row(manifest, first_source, first),
    ]
    _write_jsonl(root / "video_registration.jsonl", registration_rows)
    summary["artifacts"]["video_registration"] = str(root / "video_registration.jsonl")

    time_axis_rows = [
        _single_view_time_axis_report(third, camera_id=str(getattr(third_source, "camera_id", "") or "cam01")),
        _single_view_time_axis_report(first, camera_id=str(getattr(first_source, "camera_id", "") or "cam02")),
    ]
    time_axis_report = {
        "schema_version": "dual_view_time_axis_report.v1",
        "views": time_axis_rows,
        "status": "time_axis_unreliable"
        if any(row.get("status") == "time_axis_unreliable" for row in time_axis_rows)
        else "warning"
        if any(row.get("status") == "warning" for row in time_axis_rows)
        else "healthy",
    }
    _write_json(root / "time_axis_report.json", time_axis_report)
    summary["artifacts"]["time_axis_report"] = str(root / "time_axis_report.json")
    if time_axis_report["status"] == "time_axis_unreliable":
        _write_blocked_stage_artifacts(root, reason="time_axis_unreliable")
        summary.update(
            {
                "status": "time_axis_unreliable",
                "formal_results_allowed": False,
                "video_memory_allowed": False,
                "blocked_reason": "time_axis_unreliable",
                "elapsed_sec": round(time.perf_counter() - started, 6),
            }
        )
        _write_json(root / "dual_view_alignment_pipeline_summary.json", summary)
        return summary

    state_segments = _initial_state_scan_segments(third, first)
    _write_json(root / "state_scan_segments.json", {"schema_version": "dual_view_state_scan.v1", "segments": state_segments})
    trim_window = _trim_window_from_timelines(third, first)
    _write_json(root / "trim_window.json", trim_window)
    units = _alignment_units_from_timelines(third, first, trim_window)
    _write_json(root / "alignment_units.json", {"schema_version": "alignment_units.v1", "units": units})
    local_offsets = [
        {
            "unit_id": unit["unit_id"],
            "anchor_type": "timestamp_intersection_default",
            "third_anchor_timestamp_us": unit["third_start_timestamp_us"],
            "first_anchor_timestamp_us": unit["first_start_timestamp_us"],
            "local_offset_ms": 0.0,
            "confidence": 0.5,
            "evidence": [],
            "method": "frames_csv_common_intersection_zero_offset",
            "warnings": ["action_anchor_detection_pending_yolo"],
        }
        for unit in units
    ]
    _write_json(root / "local_offset_report.json", {"schema_version": "local_offset_report.v1", "offsets": local_offsets})

    all_samples: list[AlignmentSample] = []
    quality_units: list[dict[str, Any]] = []
    for unit in units:
        unit_id = str(unit["unit_id"])
        samples = build_alignment_samples(
            third=third,
            first=first,
            unit_id=unit_id,
            start_abs_sec=float(unit["common_start_sec"]),
            end_abs_sec=float(unit["common_end_sec"]),
            target_fps=float(target_fps or max(1.0, min(third.fps, first.fps))),
            max_pair_delta_sec=max(0.001, float(max_delta_ms) / 1000.0),
        )
        _, reference_times = reference_times_for_common_range(
            third=third,
            first=first,
            start_abs_sec=float(unit["common_start_sec"]),
            end_abs_sec=float(unit["common_end_sec"]),
            target_fps=float(target_fps or max(1.0, min(third.fps, first.fps))),
        )
        quality = alignment_quality_report(
            third=third,
            first=first,
            samples=samples,
            target_fps=float(target_fps or max(1.0, min(third.fps, first.fps))),
            source_sample_count=len(reference_times),
            max_pair_delta_sec=max(0.001, float(max_delta_ms) / 1000.0),
            output_paths={},
        )
        quality.update(
            {
                "unit_id": unit_id,
                "matched_frame_count": len(samples),
                "dropped_frame_count": max(0, len(reference_times) - len(samples)),
                "duplicated_frame_count": 0,
                "median_delta_ms": round(float((quality.get("pair_delta_sec") or {}).get("median") or 0.0) * 1000.0, 6),
                "p90_delta_ms": round(float((quality.get("pair_delta_sec") or {}).get("p90") or 0.0) * 1000.0, 6),
                "p95_delta_ms": round(_percentile([sample.pair_delta_sec * 1000.0 for sample in samples], 0.95), 6),
                "p99_delta_ms": round(_percentile([sample.pair_delta_sec * 1000.0 for sample in samples], 0.99), 6),
                "max_delta_ms": round(float((quality.get("pair_delta_sec") or {}).get("max") or 0.0) * 1000.0, 6),
                "local_offset_ms": 0.0,
                "reference_camera": quality.get("base_view"),
                "unit_status": "passed"
                if samples
                and float((quality.get("pair_delta_sec") or {}).get("median") or 0.0) * 1000.0 < float(median_gate_ms)
                and float((quality.get("pair_delta_sec") or {}).get("p90") or 0.0) * 1000.0 < float(p90_gate_ms)
                and float((quality.get("pair_delta_sec") or {}).get("max") or 0.0) * 1000.0 <= float(max_delta_ms)
                else "frame_alignment_unreliable",
                "action_phase_consistency": "pending_yolo",
                "warnings": [],
            }
        )
        quality_units.append(quality)
        all_samples.extend(samples)

    write_alignment_index(root / "sync_index.csv", all_samples, third=third, first=first, unit_id="combined")
    write_alignment_index(root / "sync_index.jsonl", all_samples, third=third, first=first, unit_id="combined")
    summary["artifacts"]["sync_index_csv"] = str(root / "sync_index.csv")
    summary["artifacts"]["sync_index_jsonl"] = str(root / "sync_index.jsonl")

    aligned_paths: dict[str, str] = {}
    if make_aligned_videos and all_samples:
        videos_quality = build_aligned_dual_view_videos(
            third_video=third.video_path,
            first_video=first.video_path,
            output_dir=root,
            third_frames_csv=third.frames_csv_path,
            first_frames_csv=first.frames_csv_path,
            target_fps=target_fps,
            max_pair_delta_sec=max(0.001, float(max_delta_ms) / 1000.0),
            segment_name="aligned",
        )
        aligned_paths = dict(videos_quality.get("output_paths") or {})
    else:
        aligned_paths = {
            "status": "not_built",
            "reason": "make_aligned_videos_disabled" if not make_aligned_videos else "no_valid_sync_pairs",
        }
    _write_json(root / "aligned_video_outputs.json", aligned_paths)

    alignment_quality = {
        "schema_version": "alignment_quality_report.v1",
        "units": quality_units,
        "status": "passed" if quality_units and all(unit.get("unit_status") == "passed" for unit in quality_units) else "frame_alignment_unreliable",
        "gate": {
            "median_gate_ms": float(median_gate_ms),
            "p90_gate_ms": float(p90_gate_ms),
            "max_delta_ms": float(max_delta_ms),
        },
    }
    _write_json(root / "alignment_quality_report.json", alignment_quality)
    phase_report = {
        "schema_version": "phase_consistency_report.v1",
        "units": [
            {
                "unit_id": unit["unit_id"],
                "consistency_score": None,
                "inconsistent_ranges": [],
                "third_phase": "unknown",
                "first_phase": "unknown",
                "issue_type": "pending_yolo_action_phase_validation",
                "status": "pending_yolo",
                "warnings": ["phase consistency is finalized after dual-view YOLO action alignment"],
            }
            for unit in units
        ],
        "status": "pending_yolo",
    }
    _write_json(root / "phase_consistency_report.json", phase_report)
    formal_windows = {
        "schema_version": "formal_experiment_windows.v1",
        "windows": [
            {
                "experiment_window_id": f"{unit['unit_id']}_candidate",
                "unit_id": unit["unit_id"],
                "start_global_timestamp_us": unit["third_start_timestamp_us"],
                "end_global_timestamp_us": unit["third_end_timestamp_us"],
                "start_sync_index": 0,
                "end_sync_index": max(0, len(all_samples) - 1),
                "start_reason": "alignment_unit_candidate_pending_yolo_phase",
                "end_reason": "alignment_unit_candidate_pending_yolo_phase",
                "confidence": 0.5,
                "status": "candidate_pending_yolo_phase",
            }
            for unit in units
            if alignment_quality["status"] == "passed"
        ],
    }
    _write_json(root / "formal_experiment_windows.json", formal_windows)
    publish_gate = {
        "schema_version": "publish_gate_report.v1",
        "gate_status": "pending_yolo_material_generation" if alignment_quality["status"] == "passed" else "rejected",
        "reject_reasons": [] if alignment_quality["status"] == "passed" else ["frame_alignment_unreliable"],
        "warnings": ["material-level gates are evaluated in material_references.py"],
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(root / "publish_gate_report.json", publish_gate)

    allowed = alignment_quality["status"] == "passed"
    summary.update(
        {
            "status": "alignment_ready_pending_yolo_phase" if allowed else "frame_alignment_unreliable",
            "pre_yolo_analysis_allowed": bool(allowed),
            "formal_results_allowed": False,
            "video_memory_allowed": False,
            "formal_results_pending_reason": "pending_action_phase_consistency" if allowed else "frame_alignment_unreliable",
            "alignment_quality_status": alignment_quality["status"],
            "phase_consistency_status": phase_report["status"],
            "aligned_video_outputs": aligned_paths,
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    )
    for artifact_name in (
        "state_scan_segments.json",
        "trim_window.json",
        "alignment_units.json",
        "local_offset_report.json",
        "alignment_quality_report.json",
        "phase_consistency_report.json",
        "formal_experiment_windows.json",
        "publish_gate_report.json",
        "aligned_video_outputs.json",
    ):
        summary["artifacts"][Path(artifact_name).stem] = str(root / artifact_name)
    _write_json(root / "dual_view_alignment_pipeline_summary.json", summary)
    return summary


def load_view_timeline(
    role: str,
    video_path: str | Path,
    frames_csv_path: str | Path | None = None,
    *,
    timestamp_field: str | None = None,
) -> ViewFrameTimeline:
    video = Path(video_path)
    frames = Path(frames_csv_path) if frames_csv_path else video.parent / "frames.csv"
    duration = _safe_ffprobe_float(video, "format=duration")
    fps = _ffprobe_video_fps(video) or 30.0
    if frames.exists():
        frame_rows = _read_rgb_frame_rows(frames, timestamp_field=timestamp_field)
        if len(frame_rows["timestamps_sec"]) >= 2:
            absolute_times = tuple(frame_rows["timestamps_sec"])
            frame_indices = tuple(frame_rows["frame_indices"])
            actual_fps = max(0.001, float(len(absolute_times) - 1) / max(0.001, float(absolute_times[-1]) - float(absolute_times[0])))
            local_secs = tuple(float(index) / max(0.001, fps) for index in frame_indices)
            return ViewFrameTimeline(
                role=role,
                video_path=str(video),
                frames_csv_path=str(frames),
                duration_sec=duration,
                fps=actual_fps,
                absolute_times_sec=absolute_times,
                local_video_secs=local_secs,
                source="frames_csv_rgb_nearest_neighbor",
                frame_indices=frame_indices,
                timestamp_field=str(frame_rows.get("timestamp_field") or ""),
                duplicate_timestamp_count=int(frame_rows.get("duplicate_timestamp_count") or 0),
                non_monotonic_timestamp_count=int(frame_rows.get("non_monotonic_timestamp_count") or 0),
                invalid_frame_count=int(frame_rows.get("invalid_frame_count") or 0),
            )
    count = max(2, int(math.floor(duration * fps)) + 1)
    times = tuple(index / fps for index in range(count))
    return ViewFrameTimeline(
        role=role,
        video_path=str(video),
        frames_csv_path=str(frames) if frames.exists() else None,
        duration_sec=duration,
        fps=fps,
        absolute_times_sec=times,
        local_video_secs=times,
        source="synthetic_pts_fallback",
        frame_indices=tuple(range(count)),
    )


def build_alignment_samples(
    *,
    third: ViewFrameTimeline,
    first: ViewFrameTimeline,
    unit_id: str = "unit_000001",
    start_abs_sec: float,
    end_abs_sec: float,
    target_fps: float,
    max_pair_delta_sec: float | None = 0.150,
) -> list[AlignmentSample]:
    samples: list[AlignmentSample] = []
    _, base_times = reference_times_for_common_range(
        third=third,
        first=first,
        start_abs_sec=start_abs_sec,
        end_abs_sec=end_abs_sec,
        target_fps=target_fps,
    )
    for index, absolute_time in enumerate(base_times):
        third_idx, third_local, third_delta = third.nearest(absolute_time)
        first_idx, first_local, first_delta = first.nearest(absolute_time)
        third_abs = third.absolute_times_sec[third_idx] if third.absolute_times_sec else absolute_time
        first_abs = first.absolute_times_sec[first_idx] if first.absolute_times_sec else absolute_time
        pair_delta = abs(float(third_abs) - float(first_abs))
        if max_pair_delta_sec is not None and pair_delta > float(max_pair_delta_sec):
            continue
        samples.append(
            AlignmentSample(
                sample_index=index,
                common_time_sec=round(absolute_time - max(third.start_abs_sec, first.start_abs_sec), 6),
                absolute_time_sec=round(absolute_time, 6),
                third_frame_index=third_idx,
                first_frame_index=first_idx,
                third_local_sec=round(third_local, 6),
                first_local_sec=round(first_local, 6),
                third_delta_sec=round(third_delta, 6),
                first_delta_sec=round(first_delta, 6),
                pair_delta_sec=round(pair_delta, 6),
            )
        )
    return samples


def reference_times_for_common_range(
    *,
    third: ViewFrameTimeline,
    first: ViewFrameTimeline,
    start_abs_sec: float,
    end_abs_sec: float,
    target_fps: float | None = None,
) -> tuple[ViewFrameTimeline, list[float]]:
    """Return the sparser real timestamp sequence inside the common range."""

    third_times = [
        float(item)
        for item in third.absolute_times_sec
        if float(start_abs_sec) <= float(item) < float(end_abs_sec)
    ]
    first_times = [
        float(item)
        for item in first.absolute_times_sec
        if float(start_abs_sec) <= float(item) < float(end_abs_sec)
    ]
    if len(third_times) and (len(third_times) <= len(first_times) or not first_times):
        base = third
        base_times = third_times
    elif first_times:
        base = first
        base_times = first_times
    else:
        base = third if third.fps <= first.fps else first
        step = 1.0 / max(0.001, float(target_fps or min(third.fps, first.fps, 15.0)))
        count = max(1, int(math.floor((end_abs_sec - start_abs_sec) / step)))
        base_times = [float(start_abs_sec) + index * step for index in range(count)]

    if target_fps is not None and base_times:
        max_fps = max(0.001, float(target_fps))
        observed_fps = max(0.001, float(len(base_times) - 1) / max(0.001, float(base_times[-1]) - float(base_times[0]))) if len(base_times) > 1 else max_fps
        if max_fps < observed_fps:
            step = 1.0 / max_fps
            selected: list[float] = []
            next_time = float(base_times[0])
            for absolute_time in base_times:
                if absolute_time + 1e-9 >= next_time:
                    selected.append(float(absolute_time))
                    next_time = float(absolute_time) + step
            base_times = selected
    return base, base_times


def alignment_quality_report(
    *,
    third: ViewFrameTimeline,
    first: ViewFrameTimeline,
    samples: Sequence[AlignmentSample],
    target_fps: float,
    source_sample_count: int | None = None,
    max_pair_delta_sec: float | None = None,
    output_paths: dict[str, str],
) -> dict[str, Any]:
    pair_deltas = [float(sample.pair_delta_sec) for sample in samples]
    third_deltas = [float(sample.third_delta_sec) for sample in samples]
    first_deltas = [float(sample.first_delta_sec) for sample in samples]
    p90 = _percentile(pair_deltas, 0.90)
    max_delta = max(pair_deltas or [0.0])
    source_count = int(source_sample_count if source_sample_count is not None else len(samples))
    dropped = max(0, source_count - len(samples))
    kept_ratio = (float(len(samples)) / float(source_count)) if source_count else 0.0
    status = "aligned_frame_time_reliable" if p90 <= 0.150 and max_delta <= 0.300 and kept_ratio >= 0.98 else "frame_time_alignment_warning"
    if not samples or max_delta > 1.0 or p90 > 0.5 or kept_ratio < 0.95:
        status = "frame_time_alignment_unreliable"
    return {
        "schema_version": "dual_view_nearest_neighbor_alignment.v1",
        "status": status,
        "note": "The lower-FPS view is used as the base frame sequence; the other view is nearest-neighbor matched. Unmatched/out-of-common-range frames are discarded. Formal output still requires action-phase validation.",
        "base_view": third.role if third.fps <= first.fps else first.role,
        "target_fps": round(float(target_fps), 6),
        "sample_count": len(samples),
        "reference_sample_count": source_count,
        "dropped_unpaired_sample_count": dropped,
        "kept_sample_ratio": round(kept_ratio, 6),
        "max_pair_delta_sec": round(float(max_pair_delta_sec), 6) if max_pair_delta_sec is not None else None,
        "discard_policy": "keep_only_common_range_base_frames_with_nearest_neighbor_pair",
        "third_person": _timeline_summary(third),
        "first_person": _timeline_summary(first),
        "pair_delta_sec": {
            "median": round(_percentile(pair_deltas, 0.50), 6),
            "p90": round(p90, 6),
            "max": round(max_delta, 6),
        },
        "third_nearest_delta_sec": {
            "median": round(_percentile(third_deltas, 0.50), 6),
            "p90": round(_percentile(third_deltas, 0.90), 6),
            "max": round(max(third_deltas or [0.0]), 6),
        },
        "first_nearest_delta_sec": {
            "median": round(_percentile(first_deltas, 0.50), 6),
            "p90": round(_percentile(first_deltas, 0.90), 6),
            "max": round(max(first_deltas or [0.0]), 6),
        },
        "output_paths": output_paths,
    }


def write_alignment_manifest(
    path: str | Path,
    third: ViewFrameTimeline,
    first: ViewFrameTimeline,
    samples: Sequence[AlignmentSample],
    quality: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "dual_view_alignment_manifest.v1",
        "third_person": _timeline_summary(third),
        "first_person": _timeline_summary(first),
        "quality": quality,
        "samples_preview": [asdict(item) for item in samples[:20]],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_alignment_index(
    path: str | Path,
    samples: Sequence[AlignmentSample],
    *,
    third: ViewFrameTimeline,
    first: ViewFrameTimeline,
    unit_id: str,
    local_offset_ms: float = 0.0,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [_sync_index_row(item, third=third, first=first, unit_id=unit_id, local_offset_ms=local_offset_ms) for item in samples]
    if output.suffix.lower() == ".jsonl":
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return
    fieldnames = [
        "sync_index",
        "unit_id",
        "global_timestamp_us",
        "reference_camera",
        "reference_frame_index",
        "reference_timestamp_us",
        "third_frame_index",
        "third_timestamp_us",
        "third_video_path",
        "first_frame_index",
        "first_timestamp_us",
        "first_video_path",
        "local_offset_ms",
        "delta_ms",
        "sync_quality",
        "is_valid_pair",
        "drop_reason",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sync_index_row(
    sample: AlignmentSample,
    *,
    third: ViewFrameTimeline,
    first: ViewFrameTimeline,
    unit_id: str,
    local_offset_ms: float,
) -> dict[str, Any]:
    reference_is_third = third.fps <= first.fps
    reference_camera = "third_person" if reference_is_third else "first_person"
    reference_frame_index = sample.third_frame_index if reference_is_third else sample.first_frame_index
    reference_timestamp_us = int(round(float(sample.absolute_time_sec) * 1_000_000.0))
    delta_ms = float(sample.pair_delta_sec) * 1000.0
    if delta_ms <= 50.0:
        quality = "good"
    elif delta_ms <= 150.0:
        quality = "acceptable"
    elif delta_ms <= 300.0:
        quality = "weak"
    else:
        quality = "drop"
    return {
        "sync_index": int(sample.sample_index),
        "unit_id": unit_id,
        "global_timestamp_us": reference_timestamp_us,
        "reference_camera": reference_camera,
        "reference_frame_index": int(reference_frame_index),
        "reference_timestamp_us": reference_timestamp_us,
        "third_frame_index": int(sample.third_frame_index),
        "third_timestamp_us": int(round((float(sample.absolute_time_sec) + float(sample.third_delta_sec)) * 1_000_000.0)),
        "third_video_path": third.video_path,
        "first_frame_index": int(sample.first_frame_index),
        "first_timestamp_us": int(round((float(sample.absolute_time_sec) + float(sample.first_delta_sec)) * 1_000_000.0)),
        "first_video_path": first.video_path,
        "local_offset_ms": round(float(local_offset_ms), 6),
        "delta_ms": round(delta_ms, 6),
        "sync_quality": quality,
        "is_valid_pair": quality != "drop",
        "drop_reason": "" if quality != "drop" else "delta_ms_exceeds_threshold",
    }


def _source_frames_csv_path(source: Any) -> str | None:
    value = getattr(source, "frames_csv_path", None)
    if value:
        return str(value)
    path = getattr(source, "path", None)
    if path:
        candidate = Path(str(path)).parent / "frames.csv"
        if candidate.exists():
            return str(candidate)
    return None


def _write_json(path: str | Path, payload: Mapping[str, Any] | dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def _video_registration_row(manifest: Any, source: Any, timeline: ViewFrameTimeline) -> dict[str, Any]:
    video_path = Path(str(getattr(source, "path", timeline.video_path)))
    stat = video_path.stat() if video_path.exists() else None
    return {
        "schema_version": "video_registration.v1",
        "video_id": _stable_video_id(video_path),
        "experiment_id": str(getattr(manifest, "session_id", "") or getattr(manifest, "run_id", "") or ""),
        "run_id": str(getattr(manifest, "session_id", "") or ""),
        "raw_video_path": str(video_path),
        "frames_csv_path": timeline.frames_csv_path,
        "camera_id": str(getattr(source, "camera_id", "") or ""),
        "view_role": timeline.role,
        "fps_metadata": float(_ffprobe_video_fps(video_path) or 0.0),
        "width": None,
        "height": None,
        "video_duration_metadata_s": float(timeline.duration_sec),
        "frame_count_metadata": len(timeline.absolute_times_sec),
        "capture_start_time": str(getattr(source, "capture_start_time", "") or getattr(source, "start_time", "") or ""),
        "timestamp_field": timeline.timestamp_field,
        "sha256": _quick_file_sha256(video_path),
        "file_size": int(stat.st_size) if stat else 0,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }


def _single_view_time_axis_report(timeline: ViewFrameTimeline, *, camera_id: str) -> dict[str, Any]:
    timestamps = [int(round(value * 1_000_000.0)) for value in timeline.absolute_times_sec]
    intervals_ms = [
        (right - left) / 1000.0
        for left, right in zip(timestamps, timestamps[1:])
        if right > left
    ]
    timestamp_duration_s = (timestamps[-1] - timestamps[0]) / 1_000_000.0 if len(timestamps) >= 2 else 0.0
    actual_fps = (float(len(timestamps) - 1) / timestamp_duration_s) if timestamp_duration_s > 0 and len(timestamps) >= 2 else 0.0
    duration_gap = abs(float(timeline.duration_sec) - float(timestamp_duration_s)) if timeline.duration_sec else 0.0
    warnings: list[str] = []
    if timeline.duplicate_timestamp_count:
        warnings.append("duplicate_timestamp_filtered")
    if timeline.non_monotonic_timestamp_count:
        warnings.append("non_monotonic_timestamp_filtered")
    large_gap_count = len([value for value in intervals_ms if value > 5000.0])
    if large_gap_count:
        warnings.append("large_timestamp_gaps_detected")
    if timeline.duration_sec and duration_gap > max(30.0, 0.02 * float(timeline.duration_sec)):
        warnings.append("mp4_duration_and_frames_csv_duration_mismatch")
    status = "time_axis_unreliable" if any("mismatch" in item for item in warnings) or any(value > 30000.0 for value in intervals_ms) else "warning" if warnings else "healthy"
    return {
        "camera_id": camera_id,
        "view_role": timeline.role,
        "video_path": timeline.video_path,
        "frames_csv_path": timeline.frames_csv_path,
        "timestamp_field": timeline.timestamp_field,
        "metadata_duration_s": round(float(timeline.duration_sec), 6),
        "timestamp_duration_s": round(timestamp_duration_s, 6),
        "duration_gap_s": round(duration_gap, 6),
        "frame_count_raw": len(timestamps) + int(timeline.duplicate_timestamp_count) + int(timeline.non_monotonic_timestamp_count) + int(timeline.invalid_frame_count),
        "frame_count_valid": len(timestamps),
        "actual_fps": round(actual_fps, 6),
        "avg_frame_interval_ms": round(sum(intervals_ms) / len(intervals_ms), 6) if intervals_ms else 0.0,
        "p50_frame_interval_ms": round(_percentile(intervals_ms, 0.50), 6),
        "p90_frame_interval_ms": round(_percentile(intervals_ms, 0.90), 6),
        "p95_frame_interval_ms": round(_percentile(intervals_ms, 0.95), 6),
        "p99_frame_interval_ms": round(_percentile(intervals_ms, 0.99), 6),
        "max_frame_interval_ms": round(max(intervals_ms or [0.0]), 6),
        "large_gap_count": large_gap_count,
        "duplicate_timestamp_count": int(timeline.duplicate_timestamp_count),
        "non_monotonic_timestamp_count": int(timeline.non_monotonic_timestamp_count),
        "pts_out_of_range_count": 0,
        "status": status,
        "warnings": warnings,
    }


def _initial_state_scan_segments(third: ViewFrameTimeline, first: ViewFrameTimeline) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timeline in (third, first):
        rows.append(
            {
                "segment_id": f"{timeline.role}_state_000001",
                "camera_id": timeline.role,
                "view_role": timeline.role,
                "start_timestamp_us": int(round(timeline.start_abs_sec * 1_000_000.0)),
                "end_timestamp_us": int(round(timeline.end_abs_sec * 1_000_000.0)),
                "state_label": "unknown",
                "confidence": 0.2,
                "evidence_frame_indices": [
                    int(timeline.frame_indices[0]) if timeline.frame_indices else 0,
                    int(timeline.frame_indices[-1]) if timeline.frame_indices else 0,
                ],
                "evidence_timestamps_us": [
                    int(round(timeline.start_abs_sec * 1_000_000.0)),
                    int(round(timeline.end_abs_sec * 1_000_000.0)),
                ],
                "method": "rule_interface_pending_yolo_state_classifier",
                "warnings": ["state labels are finalized by downstream YOLO/VLM state classifier when available"],
            }
        )
    return rows


def _trim_window_from_timelines(third: ViewFrameTimeline, first: ViewFrameTimeline) -> dict[str, Any]:
    return {
        "schema_version": "trim_window.v1",
        "windows": [
            _trim_window_for_view(third),
            _trim_window_for_view(first),
        ],
        "policy": "do_not_trim_real_experiment_actions_without_explicit_state_evidence",
    }


def _trim_window_for_view(timeline: ViewFrameTimeline) -> dict[str, Any]:
    return {
        "camera_id": timeline.role,
        "view_role": timeline.role,
        "original_start_timestamp_us": int(round(timeline.start_abs_sec * 1_000_000.0)),
        "original_end_timestamp_us": int(round(timeline.end_abs_sec * 1_000_000.0)),
        "trimmed_start_timestamp_us": int(round(timeline.start_abs_sec * 1_000_000.0)),
        "trimmed_end_timestamp_us": int(round(timeline.end_abs_sec * 1_000_000.0)),
        "trim_reason_start": "no_confident_non_experiment_start_trim",
        "trim_reason_end": "no_confident_non_experiment_end_trim",
        "protected_experiment_ranges": [],
        "warnings": ["kept full range because no confident empty/prep/ending classifier is available"],
    }


def _alignment_units_from_timelines(third: ViewFrameTimeline, first: ViewFrameTimeline, trim_window: Mapping[str, Any]) -> list[dict[str, Any]]:
    common_start = max(third.start_abs_sec, first.start_abs_sec)
    common_end = min(third.end_abs_sec, first.end_abs_sec)
    if common_end <= common_start:
        return []
    return [
        {
            "unit_id": "unit_000001",
            "unit_index": 1,
            "third_start_timestamp_us": int(round(common_start * 1_000_000.0)),
            "third_end_timestamp_us": int(round(common_end * 1_000_000.0)),
            "first_start_timestamp_us": int(round(common_start * 1_000_000.0)),
            "first_end_timestamp_us": int(round(common_end * 1_000_000.0)),
            "common_start_sec": float(common_start),
            "common_end_sec": float(common_end),
            "reason": "common_timestamp_intersection",
            "state_phase": "unknown_pending_yolo",
            "has_large_gap": False,
            "status": "candidate",
        }
    ]


def _write_blocked_stage_artifacts(root: Path, *, reason: str) -> None:
    _write_json(root / "state_scan_segments.json", {"schema_version": "dual_view_state_scan.v1", "segments": [], "blocked_reason": reason})
    _write_json(root / "trim_window.json", {"schema_version": "trim_window.v1", "windows": [], "blocked_reason": reason})
    _write_json(root / "alignment_units.json", {"schema_version": "alignment_units.v1", "units": [], "blocked_reason": reason})
    _write_json(root / "local_offset_report.json", {"schema_version": "local_offset_report.v1", "offsets": [], "blocked_reason": reason})
    _write_json(root / "alignment_quality_report.json", {"schema_version": "alignment_quality_report.v1", "units": [], "status": "blocked", "blocked_reason": reason})
    _write_json(root / "phase_consistency_report.json", {"schema_version": "phase_consistency_report.v1", "units": [], "status": "blocked", "blocked_reason": reason})
    _write_json(root / "formal_experiment_windows.json", {"schema_version": "formal_experiment_windows.v1", "windows": [], "blocked_reason": reason})
    _write_json(root / "publish_gate_report.json", {"schema_version": "publish_gate_report.v1", "gate_status": "rejected", "reject_reasons": [reason]})
    (root / "sync_index.csv").write_text("sync_index,unit_id,global_timestamp_us\n", encoding="utf-8-sig")
    (root / "sync_index.jsonl").write_text("", encoding="utf-8")


def _stable_video_id(video_path: Path) -> str:
    return hashlib.sha256(str(video_path).lower().encode("utf-8")).hexdigest()[:16]


def _quick_file_sha256(video_path: Path) -> str | None:
    if not video_path.exists() or not video_path.is_file():
        return None
    digest = hashlib.sha256()
    stat = video_path.stat()
    with video_path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()


def _extract_view_frames(video_path: str, seconds: Sequence[float], out_dir: Path, max_width: int) -> dict[str, Any]:
    import cv2

    started = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    written = 0
    last_frame = None
    for index, second in enumerate(seconds):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(second)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            if last_frame is None:
                continue
            frame = last_frame
        last_frame = frame
        h, w = frame.shape[:2]
        if w > max_width:
            scale = float(max_width) / float(w)
            frame = cv2.resize(frame, (max_width, max(2, int(h * scale))))
        cv2.imwrite(str(out_dir / f"frame_{index:06d}.jpg"), frame)
        written += 1
    cap.release()
    return {"written": written, "elapsed_sec": time.perf_counter() - started}


def _extract_view_frames_by_index(video_path: str, frame_indices: Sequence[int], out_dir: Path, max_width: int) -> dict[str, Any]:
    import cv2

    started = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    written = 0
    last_frame = None
    requested = [max(0, int(frame_index)) for frame_index in frame_indices]
    is_monotonic = all(next_index >= current for current, next_index in zip(requested, requested[1:]))

    if is_monotonic:
        target_iter = iter(enumerate(requested))
        try:
            output_index, target_frame = next(target_iter)
        except StopIteration:
            cap.release()
            return {"written": 0, "elapsed_sec": time.perf_counter() - started, "read_mode": "sequential"}
        current_frame = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            while current_frame == target_frame:
                last_frame = frame
                h, w = frame.shape[:2]
                output_frame = frame
                if w > max_width:
                    scale = float(max_width) / float(w)
                    output_frame = cv2.resize(output_frame, (max_width, max(2, int(h * scale))))
                cv2.imwrite(str(out_dir / f"frame_{output_index:06d}.jpg"), output_frame)
                written += 1
                try:
                    output_index, target_frame = next(target_iter)
                except StopIteration:
                    cap.release()
                    return {
                        "written": written,
                        "elapsed_sec": time.perf_counter() - started,
                        "read_mode": "sequential",
                    }
                if target_frame < current_frame and last_frame is not None:
                    h, w = last_frame.shape[:2]
                    output_frame = last_frame
                    if w > max_width:
                        scale = float(max_width) / float(w)
                        output_frame = cv2.resize(output_frame, (max_width, max(2, int(h * scale))))
                    cv2.imwrite(str(out_dir / f"frame_{output_index:06d}.jpg"), output_frame)
                    written += 1
                    try:
                        output_index, target_frame = next(target_iter)
                    except StopIteration:
                        cap.release()
                        return {
                            "written": written,
                            "elapsed_sec": time.perf_counter() - started,
                            "read_mode": "sequential",
                        }
            current_frame += 1
        cap.release()
        return {"written": written, "elapsed_sec": time.perf_counter() - started, "read_mode": "sequential"}

    for index, frame_index in enumerate(requested):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            if last_frame is None:
                continue
            frame = last_frame
        last_frame = frame
        h, w = frame.shape[:2]
        if w > max_width:
            scale = float(max_width) / float(w)
            frame = cv2.resize(frame, (max_width, max(2, int(h * scale))))
        cv2.imwrite(str(out_dir / f"frame_{index:06d}.jpg"), frame)
        written += 1
    cap.release()
    return {"written": written, "elapsed_sec": time.perf_counter() - started, "read_mode": "random_seek"}


def _encode_frames(frame_dir: Path, output_path: Path, fps: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{float(fps):.6f}",
        "-i",
        str(frame_dir / "frame_%06d.jpg"),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _encode_side_by_side(third_video: Path, first_video: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(third_video),
        "-i",
        str(first_video),
        "-filter_complex",
        "[0:v]scale=640:-2,setsar=1[l];[1:v]scale=640:-2,setsar=1[r];[l][r]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _read_rgb_times(frames_csv: Path) -> list[float]:
    return list(_read_rgb_frame_rows(frames_csv).get("timestamps_sec") or [])


def _read_rgb_frame_rows(frames_csv: Path, *, timestamp_field: str | None = None) -> dict[str, Any]:
    values: list[float] = []
    frame_indices: list[int] = []
    duplicate_count = 0
    non_monotonic_count = 0
    invalid_count = 0
    seen: set[int] = set()
    last_timestamp: int | None = None
    with frames_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        candidates = (
            [timestamp_field]
            if timestamp_field
            else [
                "best_timestamp_us",
                "hardware_trigger_timestamp_us",
                "ptp_timestamp_us",
                "camera_timestamp_us",
                "packet_system_timestamp_us",
                "local_time_us",
                "rgb_local_time_us",
                "timestamp_us",
                "frame_system_timestamp_us",
            ]
        )
        timestamp_column = next((name for name in candidates if name and name in fields), None)
        if timestamp_column is None:
            return {
                "timestamps_sec": [],
                "frame_indices": [],
                "timestamp_field": timestamp_field,
                "duplicate_timestamp_count": 0,
                "non_monotonic_timestamp_count": 0,
                "invalid_frame_count": 0,
                "error": "timestamp_field_missing",
            }
        stream_column = "stream_type" if "stream_type" in fields else None
        frame_column = next((name for name in ("frame_index", "frame_id", "rgb_frame_index", "index") if name in fields), None)
        for ordinal, row in enumerate(reader):
            if stream_column and str(row.get(stream_column) or "").strip().lower() != "rgb":
                continue
            if _false_like(row.get("read_success")) or _false_like(row.get("write_success")) or _false_like(row.get("valid")):
                invalid_count += 1
                continue
            try:
                raw_ts = int(float(str(row[timestamp_column]).replace("_", "")))
            except (TypeError, ValueError):
                invalid_count += 1
                continue
            if raw_ts in seen:
                duplicate_count += 1
                continue
            if last_timestamp is not None and raw_ts <= last_timestamp:
                non_monotonic_count += 1
                continue
            seen.add(raw_ts)
            last_timestamp = raw_ts
            frame_idx = ordinal
            if frame_column:
                try:
                    frame_idx = int(float(str(row.get(frame_column)).replace("_", "")))
                except (TypeError, ValueError):
                    invalid_count += 1
                    continue
            values.append(float(raw_ts) / 1_000_000.0)
            frame_indices.append(frame_idx)
    return {
        "timestamps_sec": values,
        "frame_indices": frame_indices,
        "timestamp_field": timestamp_column,
        "duplicate_timestamp_count": duplicate_count,
        "non_monotonic_timestamp_count": non_monotonic_count,
        "invalid_frame_count": invalid_count,
    }


def _false_like(value: Any) -> bool:
    return str(value).strip().lower() in {"0", "false", "no", "n", "failed", "invalid"}


def _safe_ffprobe_float(video_path: Path, entry: str) -> float:
    try:
        return _ffprobe_float(video_path, entry)
    except Exception:
        return 0.0


def _ffprobe_float(video_path: Path, entry: str) -> float:
    payload = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", entry, "-of", "json", str(video_path)],
        text=True,
    )
    data = json.loads(payload)
    if entry == "format=duration":
        return float(data["format"]["duration"])
    raise ValueError(entry)


def _ffprobe_video_fps(video_path: Path) -> float | None:
    try:
        payload = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "json", str(video_path)],
            text=True,
        )
        rate = json.loads(payload)["streams"][0]["avg_frame_rate"]
        num, den = rate.split("/")
        den_f = float(den)
        return float(num) / den_f if den_f else None
    except Exception:
        return None


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * min(max(float(q), 0.0), 1.0)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timeline_summary(timeline: ViewFrameTimeline) -> dict[str, Any]:
    return {
        "role": timeline.role,
        "video_path": timeline.video_path,
        "frames_csv_path": timeline.frames_csv_path,
        "duration_sec": round(float(timeline.duration_sec), 6),
        "fps": round(float(timeline.fps), 6),
        "frame_count": len(timeline.absolute_times_sec),
        "start_abs_sec": round(float(timeline.start_abs_sec), 6),
        "end_abs_sec": round(float(timeline.end_abs_sec), 6),
        "source": timeline.source,
    }

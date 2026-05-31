"""Benchmark SessionManifest construction + detection-config validation (P0.1).

This module builds a valid ``SessionManifest`` for a dual-view dataset and
validates the detection config BEFORE any GPU run, so a 3-hour job is never
launched against a misconfigured manifest. It performs no GPU work and decodes
no video; it only reads frames.csv headers and meta.json.

Key safety checks it enforces:
  * YOLO weights actually exist on disk (per view).
  * The configured weights are NOT the generic COCO fallback
    (``resolve_default_yolo_model`` silently substitutes ``yolo26s.pt`` /
    ``yolov8n.pt`` when no path is given — that would detect zero lab objects).
  * ``detector_backend == "yolo"`` so the lab detector is actually used.
  * The timestamp field required for alignment is present in frames.csv.
  * Camera→view role assignment is explicit (never guessed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TIMESTAMP_FIELD = "packet_system_timestamp_us"

# Generic (non-lab) weight filenames that must never be used for a benchmark
# run. These are what resolve_default_yolo_model falls back to.
GENERIC_WEIGHT_NAMES = {
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo26s.pt", "yolo26n.pt", "yolov5su.pt",
}


@dataclass
class ViewSpec:
    """One camera's resolved inputs for manifest construction."""

    role: str  # "first_person" | "third_person"
    video_path: Path
    frames_csv_path: Path
    meta_path: Path | None = None
    camera_id: str | None = None

    def meta(self) -> dict[str, Any]:
        if self.meta_path and self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return {}


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = handle.readline().strip()
    return [c.strip() for c in first.split(",") if c.strip()]


def _us_to_isoformat(us: int) -> str:
    """Microsecond epoch -> ISO8601 (UTC). Avoids wall-clock 'now'."""
    seconds = int(us) // 1_000_000
    micros = int(us) % 1_000_000
    # Build deterministically without datetime.now(); use UTC epoch.
    from datetime import datetime, timezone

    return (
        datetime.fromtimestamp(seconds, tz=timezone.utc)
        .replace(microsecond=micros)
        .isoformat()
    )


def discover_virtual_camera_pair(
    dataset_root: Path,
    *,
    first_person_camera: str,
    third_person_camera: str,
) -> dict[str, ViewSpec]:
    """Locate rgb.mp4 + frames.csv + meta.json under a virtual-camera dataset.

    Layout: ``<root>/<sender>_<camera>/<date>/<segment>_virtual/rgb.mp4``.
    Role assignment is explicit (caller names which camera is first/third);
    nothing is guessed from the directory name.
    """
    root = Path(dataset_root)
    specs: dict[str, ViewSpec] = {}
    mapping = {
        "first_person": first_person_camera,
        "third_person": third_person_camera,
    }
    for role, camera in mapping.items():
        matches = sorted(root.glob(f"*{camera}/**/rgb.mp4"))
        if not matches:
            raise FileNotFoundError(
                f"no rgb.mp4 found for camera={camera!r} (role={role}) under {root}"
            )
        video = matches[0]
        seg_dir = video.parent
        frames = seg_dir / "frames.csv"
        meta = seg_dir / "meta.json"
        if not frames.exists():
            raise FileNotFoundError(f"frames.csv missing beside {video}")
        specs[role] = ViewSpec(
            role=role,
            video_path=video,
            frames_csv_path=frames,
            meta_path=meta if meta.exists() else None,
            camera_id=camera,
        )
    return specs


def build_session_manifest(
    *,
    session_id: str,
    specs: dict[str, ViewSpec],
    output_dir: Path,
    first_person_weights: Path,
    third_person_weights: Path,
    coarse_sample_fps: float = 1.0,
    fine_sample_fps: float = 6.0,
    yolo_device: str = "auto",
    timestamp_field: str = DEFAULT_TIMESTAMP_FIELD,
) -> dict[str, Any]:
    """Produce a manifest dict ready for SessionManifest.from_dict.

    Weights are wired explicitly into detection_config so the COCO fallback is
    never hit. No GPU work is done here.
    """
    third = specs["third_person"]
    first = specs.get("first_person")

    def _video_block(spec: ViewSpec) -> dict[str, Any]:
        meta = spec.meta()
        seg_start = meta.get("segment_start_us")
        fps = meta.get("rgb_actual_fps") or meta.get("rgb_fps")
        block = {
            "name": spec.role,
            "path": str(spec.video_path),
            "role": spec.role,
            "camera_id": spec.camera_id,
            "frames_csv_path": str(spec.frames_csv_path),
            "start_time": _us_to_isoformat(seg_start)
            if seg_start is not None
            else "1970-01-01T00:00:00+00:00",
        }
        if fps:
            block["fps"] = float(fps)
        return block

    videos: dict[str, Any] = {"third_person": _video_block(third)}
    if first is not None:
        videos["first_person"] = _video_block(first)

    session_start = videos["third_person"]["start_time"]

    manifest: dict[str, Any] = {
        "session_id": session_id,
        "session_start_time": session_start,
        "videos": videos,
        "output_dir": str(output_dir),
        "detection_config": {
            "detector_backend": "yolo",
            "yolo_scan_both_views": True,
            "sample_fps": coarse_sample_fps,
            "micro_refine_sample_fps": fine_sample_fps,
            "yolo_device": yolo_device,
            "yolo_first_person_model_path": str(first_person_weights),
            "yolo_third_person_model_path": str(third_person_weights),
        },
        "config": {"timestamp_field": timestamp_field},
    }
    return manifest


@dataclass
class ManifestValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "benchmark_manifest_validation.v1",
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


def validate_manifest_for_benchmark(
    manifest: dict[str, Any],
    *,
    timestamp_field: str = DEFAULT_TIMESTAMP_FIELD,
) -> ManifestValidation:
    """Pre-flight validation. Reads only headers/metadata; no GPU, no decode."""
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}

    dc = manifest.get("detection_config") or {}
    videos = manifest.get("videos") or {}

    # 1. detector backend must be yolo
    if dc.get("detector_backend") != "yolo":
        errors.append(
            f"detector_backend is {dc.get('detector_backend')!r}, expected 'yolo' "
            "(a benchmark run must use the lab YOLO detector)."
        )

    # 2. weights present + not generic COCO fallback
    for role, key in (
        ("first_person", "yolo_first_person_model_path"),
        ("third_person", "yolo_third_person_model_path"),
    ):
        wpath = dc.get(key)
        if not wpath:
            # only an error if that view exists in the manifest
            if role in videos:
                errors.append(f"{key} missing — would fall back to generic COCO weights.")
            continue
        p = Path(wpath)
        if not p.is_file():
            errors.append(f"{key} -> {wpath} does not exist on disk.")
        elif p.name in GENERIC_WEIGHT_NAMES:
            errors.append(
                f"{key} -> {p.name} is a generic COCO weight, not a lab-trained model."
            )
        else:
            info[key] = str(p)

    # 3. videos + frames.csv + timestamp field
    for role in ("third_person", "first_person"):
        block = videos.get(role)
        if not block:
            if role == "third_person":
                errors.append("videos.third_person is required.")
            continue
        vpath = Path(block.get("path", ""))
        if not vpath.is_file():
            errors.append(f"videos.{role}.path -> {vpath} does not exist.")
        fcsv = block.get("frames_csv_path")
        if not fcsv:
            errors.append(f"videos.{role}.frames_csv_path missing.")
            continue
        fpath = Path(fcsv)
        if not fpath.is_file():
            errors.append(f"videos.{role}.frames_csv_path -> {fpath} does not exist.")
            continue
        header = _read_csv_header(fpath)
        if timestamp_field not in header:
            errors.append(
                f"videos.{role} frames.csv lacks timestamp field {timestamp_field!r} "
                f"(header has: {header[:6]}...)."
            )
        else:
            info.setdefault("frames_csv_columns", {})[role] = header

    # 4. config sanity (non-fatal)
    if dc.get("sample_fps") and dc.get("micro_refine_sample_fps"):
        if float(dc["micro_refine_sample_fps"]) < float(dc["sample_fps"]):
            warnings.append(
                "micro_refine_sample_fps (fine) < sample_fps (coarse); fine scan "
                "should be denser than coarse."
            )

    return ManifestValidation(
        ok=not errors, errors=errors, warnings=warnings, info=info
    )

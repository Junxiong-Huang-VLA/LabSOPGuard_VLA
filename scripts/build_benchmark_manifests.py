"""Build + validate benchmark manifests (smoke + full) for the 2026-05-22 run.

Roles confirmed by the user:
  cam01 = third_person (fixed)  -> yolo26s_third_20_v2/best.pt
  cam02 = first_person (neck)   -> yolo26_first_20_v1/yolo26_first_20_v1.pt

The smoke slice truncates each frames.csv to the first SMOKE_SECONDS of the
virtual timeline into a temp dataset (videos are still the real mp4 — decode
only seeks early frames). No GPU work happens in this script; it only writes
manifests + a truncated CSV and runs the read-only validator.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from key_action_indexer import benchmark_manifest as bm

DATASET = Path(
    r"D:\LabVideo\raw_uploads\by_import"
    r"\virtual_camera_cache_from_fullres_aligned_2026-05-22"
)
W_FIRST = Path(r"C:\Users\Xx7\Desktop\yolo26_first_20_v1\yolo26_first_20_v1.pt")
W_THIRD = Path(r"C:\Users\Xx7\Desktop\yolo26s_third_20_v2\best.pt")

OUT_ROOT = Path(r"D:\LabCapability\_benchmark")
SESSION_FULL = "benchmark-weighing-pipetting-2026-05-22"
SESSION_SMOKE = "benchmark-smoke-2026-05-22"
OUTPUTS = Path(r"D:\LabCapability\LabSOPGuard\outputs\experiments")

SMOKE_SECONDS = 120.0
TIMESTAMP_FIELD = "packet_system_timestamp_us"


def _truncate_frames_csv(src: Path, dst: Path, seconds: float) -> int:
    """Copy only rows within the first `seconds` of the virtual timeline."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        rows = []
        t0 = None
        for row in reader:
            ts = int(row[TIMESTAMP_FIELD])
            if t0 is None:
                t0 = ts
            if (ts - t0) / 1_000_000.0 > seconds:
                break
            rows.append(row)
        fieldnames = reader.fieldnames or []
    with dst.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_full() -> dict:
    specs = bm.discover_virtual_camera_pair(
        DATASET, first_person_camera="cam02", third_person_camera="cam01"
    )
    manifest = bm.build_session_manifest(
        session_id=SESSION_FULL,
        specs=specs,
        output_dir=OUTPUTS / SESSION_FULL,
        first_person_weights=W_FIRST,
        third_person_weights=W_THIRD,
        coarse_sample_fps=1.0,
        fine_sample_fps=6.0,
        yolo_device="cuda:0",
        timestamp_field=TIMESTAMP_FIELD,
    )
    return manifest


def build_smoke() -> dict:
    """Truncated dataset: real mp4 + first-120s frames.csv."""
    specs = bm.discover_virtual_camera_pair(
        DATASET, first_person_camera="cam02", third_person_camera="cam01"
    )
    smoke_specs: dict[str, bm.ViewSpec] = {}
    for role, spec in specs.items():
        seg_dir = OUT_ROOT / "smoke_dataset" / role
        seg_dir.mkdir(parents=True, exist_ok=True)
        trimmed = seg_dir / "frames.csv"
        n = _truncate_frames_csv(spec.frames_csv_path, trimmed, SMOKE_SECONDS)
        # copy meta.json so fps/start metadata resolves
        if spec.meta_path and spec.meta_path.exists():
            shutil.copy2(spec.meta_path, seg_dir / "meta.json")
        # Use a trimmed mp4 (exactly SMOKE_SECONDS long) so the probed video
        # duration matches the truncated frames.csv span. Pointing at the full
        # 10,491s mp4 would trip the time-axis health check
        # (capture_mp4_duration_delta) and is not a faithful smoke.
        trimmed_video = seg_dir / "rgb.mp4"
        if not trimmed_video.exists():
            raise FileNotFoundError(
                f"trimmed smoke video missing: {trimmed_video} "
                "(run the ffmpeg -t 120 trim step first)"
            )
        smoke_specs[role] = bm.ViewSpec(
            role=role,
            video_path=trimmed_video,
            frames_csv_path=trimmed,
            meta_path=(seg_dir / "meta.json") if (seg_dir / "meta.json").exists() else None,
            camera_id=spec.camera_id,
        )
        print(f"  smoke {role}: {n} rows (~{SMOKE_SECONDS:.0f}s) -> {trimmed}")
    manifest = bm.build_session_manifest(
        session_id=SESSION_SMOKE,
        specs=smoke_specs,
        output_dir=OUTPUTS / SESSION_SMOKE,
        first_person_weights=W_FIRST,
        third_person_weights=W_THIRD,
        coarse_sample_fps=1.0,
        fine_sample_fps=6.0,
        yolo_device="cuda:0",
        timestamp_field=TIMESTAMP_FIELD,
    )
    return manifest


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Building smoke manifest...")
    smoke = build_smoke()
    print("Building full manifest...")
    full = build_full()

    fail = False
    for name, manifest in (("smoke", smoke), ("full", full)):
        v = bm.validate_manifest_for_benchmark(manifest, timestamp_field=TIMESTAMP_FIELD)
        path = OUT_ROOT / f"manifest_{name}.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[{name}] -> {path}")
        print(f"  ok={v.ok}")
        if v.errors:
            fail = True
            for e in v.errors:
                print(f"  ERROR: {e}")
        for w in v.warnings:
            print(f"  warn: {w}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

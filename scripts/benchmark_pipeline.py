"""Benchmark script to measure pipeline performance across stages.

Usage:
    python scripts/benchmark_pipeline.py [--video PATH] [--experiment-id ID]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labsopguard.config import load_runtime_settings
from labsopguard.event_preprocessing.activity_presegmenter import ActivityPreSegmenter, PresegmentConfig
from labsopguard.event_preprocessing.engine import EventPreprocessingEngine


def benchmark_presegment(video_path: Path, project_root: Path) -> dict:
    config = PresegmentConfig()
    segmenter = ActivityPreSegmenter(config)

    start = time.perf_counter()
    segments = segmenter.segment(video_path)
    elapsed = time.perf_counter() - start

    total_active = sum(s.duration_sec for s in segments)
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps
    cap.release()

    return {
        "stage": "presegment",
        "elapsed_sec": round(elapsed, 3),
        "video_duration_sec": round(duration, 1),
        "segments_found": len(segments),
        "total_active_sec": round(total_active, 1),
        "reduction_pct": round((1 - total_active / duration) * 100, 1) if duration > 0 else 0,
        "segments": [s.to_dict() for s in segments],
    }


def benchmark_full_pipeline(video_path: Path, project_root: Path, experiment_id: str) -> dict:
    settings = load_runtime_settings(project_root)
    engine = EventPreprocessingEngine(settings)

    import tempfile
    output_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
    material_index_path = output_dir / "material_index.db"

    start = time.perf_counter()
    result = engine.run(
        experiment_id=experiment_id,
        experiment_name="benchmark_test",
        source_video=video_path,
        output_dir=output_dir,
        material_index_path=material_index_path,
    )
    elapsed = time.perf_counter() - start

    return {
        "stage": "full_pipeline",
        "elapsed_sec": round(elapsed, 3),
        "detection_frame_count": result["preprocessing_payload"]["event_preprocessing"]["detection_frame_count"],
        "event_count": result["preprocessing_payload"]["event_preprocessing"]["physical_event_count"],
        "tracklet_count": result["preprocessing_payload"]["event_preprocessing"]["tracklet_count"],
    }


def main():
    parser = argparse.ArgumentParser(description="Pipeline performance benchmark")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--experiment-id", type=str, default="benchmark-test")
    parser.add_argument("--presegment-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.video:
        video_path = Path(args.video)
    else:
        experiments_dir = project_root / "outputs" / "experiments"
        if experiments_dir.exists():
            for exp_dir in sorted(experiments_dir.iterdir(), reverse=True):
                raw_dir = exp_dir / "raw"
                if raw_dir.exists():
                    for vf in raw_dir.glob("*.mp4"):
                        video_path = vf
                        break
                    if video_path:
                        break
        if not video_path:
            print("ERROR: No video found. Use --video to specify one.")
            sys.exit(1)

    print(f"Video: {video_path}")
    print(f"Size: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    # Check GPU
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: not installed")
    print()

    # Benchmark presegment
    print("=" * 60)
    print("STAGE: Pre-segmentation")
    print("=" * 60)
    preseg_result = benchmark_presegment(video_path, project_root)
    print(f"  Time: {preseg_result['elapsed_sec']}s")
    print(f"  Video duration: {preseg_result['video_duration_sec']}s")
    print(f"  Active segments: {preseg_result['segments_found']}")
    print(f"  Active time: {preseg_result['total_active_sec']}s")
    print(f"  Reduction: {preseg_result['reduction_pct']}%")
    print()

    if not args.presegment_only:
        print("=" * 60)
        print("STAGE: Full Pipeline (presegment + YOLO + tracking + events)")
        print("=" * 60)
        pipeline_result = benchmark_full_pipeline(video_path, project_root, args.experiment_id)
        print(f"  Time: {pipeline_result['elapsed_sec']}s")
        print(f"  Frames analyzed: {pipeline_result['detection_frame_count']}")
        print(f"  Events detected: {pipeline_result['event_count']}")
        print(f"  Tracklets: {pipeline_result['tracklet_count']}")
        print()

    # Save results
    results_path = project_root / "outputs" / "benchmark_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {"presegment": preseg_result}
    if not args.presegment_only:
        all_results["full_pipeline"] = pipeline_result
    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    video_path = None
    main()

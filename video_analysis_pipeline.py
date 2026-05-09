#!/usr/bin/env python
"""Compatibility entrypoint for the formal video-analysis pipeline."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

from labsopguard import VideoAnalysisPipeline, load_runtime_settings


def _build_pipeline(sample_interval: float, max_frames: int, yolo_model_path: str | None):
    settings = load_runtime_settings(PROJECT_ROOT)
    return VideoAnalysisPipeline(
        settings=settings,
        yolo_model_path=yolo_model_path,
        vlm_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        vlm_base_url=os.environ.get("DASHSCOPE_BASE_URL"),
        sample_interval=sample_interval,
        max_frames=max_frames,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LabSOPGuard video analysis pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--yolo", help="YOLO model path")
    parser.add_argument("--output", help="Output annotated video path")
    parser.add_argument("--interval", type=float, default=3.0, help="Sample interval in seconds")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum number of frames to analyze")
    args = parser.parse_args()

    pipeline = _build_pipeline(args.interval, args.max_frames, args.yolo)
    analyses = pipeline.analyze_video(args.video)

    output_dir = PROJECT_ROOT / "outputs" / "video_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_json.write_text(
        json.dumps(pipeline.export_json(analyses), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_video = args.output or str(output_dir / f"annotated_{Path(args.video).name}")
    pipeline.create_annotated_video(args.video, analyses, output_video)

    print(output_video)
    print(output_json)


if __name__ == "__main__":
    main()

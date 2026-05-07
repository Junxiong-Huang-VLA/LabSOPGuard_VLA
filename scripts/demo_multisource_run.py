#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

from experiment.service import ExperimentService
from labsopguard.retrieval import MaterialQuery, MaterialRetrievalIndex


def _dataset_images(limit: int = 12) -> List[Path]:
    images: List[Path] = []
    for folder in [
        PROJECT_ROOT / "data" / "dataset" / "images" / "train",
        PROJECT_ROOT / "data" / "dataset" / "images" / "val",
        PROJECT_ROOT / "data" / "dataset" / "images" / "test",
    ]:
        if folder.exists():
            images.extend(folder.glob("*.jpg"))
        if len(images) >= limit:
            break
    return images[:limit]


def _make_video(images: List[Path], output_path: Path, fps: float = 4.0) -> Path:
    if not images:
        raise FileNotFoundError("No dataset images found for demo video generation.")

    def read_image(path: Path):
        frame = cv2.imread(str(path))
        if frame is None:
            data = np.fromfile(str(path), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return frame

    first = read_image(images[0])
    if first is None:
        raise RuntimeError(f"Cannot read demo image: {images[0]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")
    try:
        for image_path in images:
            frame = read_image(image_path)
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    return output_path


def run_demo(exp_id: str, max_frames: int, sample_interval: float) -> dict:
    demo_dir = PROJECT_ROOT / "outputs" / "demo_inputs" / exp_id
    primary_video = _make_video(_dataset_images(16), demo_dir / "local_camera.mp4")
    rtsp_sim_video = demo_dir / "rtsp_sim_camera.mp4"
    shutil.copyfile(primary_video, rtsp_sim_video)

    service = ExperimentService(
        frame_sample_interval=sample_interval,
        max_frames=max_frames,
    )
    service.set_video_inputs(
        [
            {
                "video_index": 0,
                "video_path": str(primary_video),
                "source_type": "file",
                "ingest_mode": "file",
                "camera_id": "cam_local",
                "sync_group": "demo_sync",
                "start_offset_sec": 0.0,
            },
            {
                "video_index": 1,
                "video_path": str(rtsp_sim_video),
                "source_type": "file",
                "ingest_mode": "rtsp_simulated_by_local_file",
                "camera_id": "cam_rtsp_sim",
                "sync_group": "demo_sync",
                "start_offset_sec": 0.35,
                "clock_drift_ppm": 10.0,
            },
        ]
    )
    service.set_context_inputs(
        [
            {
                "kind": "transcript",
                "source_type": "asr",
                "text": "0.2 seconds: operator picks up reagent bottle and prepares transfer.",
                "timestamp_sec": 0.2,
                "start_time_sec": 0.2,
                "end_time_sec": 1.2,
            },
            {
                "kind": "transcript",
                "source_type": "asr",
                "text": "1.8 seconds: sample bottle is placed near the reagent bottle.",
                "timestamp_sec": 1.8,
                "start_time_sec": 1.8,
                "end_time_sec": 2.8,
            },
        ]
    )
    service.set_protocol("Prepare reagent bottle, place sample bottle, transfer reagent, record observation.")
    result = service.process(experiment_id=exp_id, experiment_title="final acceptance multisource demo")
    paths = service.save_outputs(str(PROJECT_ROOT / "outputs" / "experiments"))
    exp_dir = Path(paths["experiment"]).parent

    material_stream = json.loads(Path(paths["material_stream"]).read_text(encoding="utf-8"))
    physical_events = json.loads(Path(paths["physical_events"]).read_text(encoding="utf-8"))
    preprocessing = json.loads(Path(paths["preprocessing"]).read_text(encoding="utf-8"))
    index = MaterialRetrievalIndex(paths["material_index"])
    try:
        search_items = index.query(MaterialQuery(objects=["reagent_bottle"], limit=10))
        health = index.health_check()
    finally:
        index.close()

    summary = {
        "experiment_id": exp_id,
        "output_dir": str(exp_dir),
        "paths": paths,
        "video_inputs": [str(primary_video), str(rtsp_sim_video)],
        "material_stream_count": len(material_stream),
        "physical_event_count": len(physical_events),
        "detected_change_count": len(preprocessing.get("detected_changes", [])),
        "key_frame_count": len(preprocessing.get("key_frames", [])),
        "key_clip_count": len(preprocessing.get("key_clips", [])),
        "search_reagent_bottle_count": len(search_items),
        "material_index_health": health,
        "models_used": result["experiment"].models_used,
    }
    (exp_dir / "acceptance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a minimal multisource acceptance experiment.")
    parser.add_argument("--exp-id", default="final_acceptance_e2e")
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    args = parser.parse_args()
    summary = run_demo(args.exp_id, args.max_frames, args.sample_interval)
    required_positive = [
        "material_stream_count",
        "key_frame_count",
        "search_reagent_bottle_count",
    ]
    return 0 if all(summary.get(key, 0) > 0 for key in required_positive) else 2


if __name__ == "__main__":
    raise SystemExit(main())

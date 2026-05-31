from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from key_action_indexer.yolo_detector import (
    HAND_LABELS,
    INTERACTION_OBJECT_LABELS,
    YoloActivityScorer,
    _detections_from_model_batch,
    _load_yolo_model,
    _row_from_detections,
    canonical_yolo_label,
)


@dataclass
class VideoSpec:
    view: str
    path: Path
    model_path: Path
    fps: float
    frames: int
    duration_sec: float
    width: int
    height: int


@dataclass
class TimingRow:
    stage: str
    view: str
    chunk_id: str
    start_sec: float
    end_sec: float
    sample_fps: float
    sampled_frames: int = 0
    decode_sec: float = 0.0
    inference_sec: float = 0.0
    postprocess_sec: float = 0.0
    write_sec: float = 0.0
    wall_sec: float = 0.0
    effective_fps: float = 0.0
    notes: list[str] = field(default_factory=list)


def _now_id() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _probe_video(path: Path, view: str, model_path: Path) -> VideoSpec:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frames / fps if fps > 0 else 0.0
        return VideoSpec(view, path, model_path, fps, frames, duration, width, height)
    finally:
        cap.release()


def _ranges(duration_sec: float, chunk_sec: float) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_sec - 1e-6:
        end = min(duration_sec, start + chunk_sec)
        ranges.append((start, end))
        start = end
    return ranges


def _target_times(start_sec: float, end_sec: float, sample_fps: float) -> list[float]:
    if end_sec <= start_sec or sample_fps <= 0:
        return []
    step = 1.0 / sample_fps
    count = int(math.floor((end_sec - start_sec) / step)) + 1
    return [min(end_sec, start_sec + index * step) for index in range(count)]


def _labels(row: dict[str, Any]) -> set[str]:
    labels = set()
    for key, value in dict(row.get("label_counts") or {}).items():
        if int(value or 0) > 0:
            labels.add(canonical_yolo_label(key))
    for det in row.get("detections") or []:
        if isinstance(det, dict):
            labels.add(canonical_yolo_label(det.get("label")))
    labels.discard("")
    return labels


def _row_evidence_score(row: dict[str, Any]) -> float:
    labels = _labels(row)
    has_hand = bool(labels & set(HAND_LABELS))
    has_object = bool(labels & set(INTERACTION_OBJECT_LABELS))
    interaction_score = max([float(item.get("score", 0.0) or 0.0) for item in row.get("hand_object_interactions") or []], default=0.0)
    if interaction_score >= 0.12:
        return min(1.0, 0.75 + interaction_score * 0.25)
    if has_hand and has_object:
        return 0.68
    if has_hand:
        return 0.35
    if has_object:
        return 0.12
    return 0.0


def _scan_window(
    *,
    spec: VideoSpec,
    model: Any,
    stage: str,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
    batch_size: int,
    conf: float,
    iou: float,
    device: str,
    imgsz: int,
    scorer: YoloActivityScorer,
    chunk_id: str,
) -> tuple[list[dict[str, Any]], TimingRow]:
    wall_start = time.perf_counter()
    timing = TimingRow(
        stage=stage,
        view=spec.view,
        chunk_id=chunk_id,
        start_sec=round(start_sec, 6),
        end_sec=round(end_sec, 6),
        sample_fps=float(sample_fps),
    )
    rows: list[dict[str, Any]] = []
    cap = cv2.VideoCapture(str(spec.path))
    if not cap.isOpened():
        timing.notes.append("open_failed")
        timing.wall_sec = round(time.perf_counter() - wall_start, 6)
        return rows, timing
    try:
        times = _target_times(start_sec, end_sec, sample_fps)
        batch_frames: list[Any] = []
        batch_meta: list[tuple[int, int, float]] = []
        sample_index_base = int(round(start_sec * max(sample_fps, 1e-6)))

        def flush() -> None:
            if not batch_frames:
                return
            infer_start = time.perf_counter()
            detections_batch = _detections_from_model_batch(
                model,
                batch_frames,
                conf=conf,
                iou=iou,
                device=device,
                imgsz=imgsz,
            )
            timing.inference_sec += time.perf_counter() - infer_start
            post_start = time.perf_counter()
            for (frame_index, sample_index, time_sec), frame, detections in zip(batch_meta, batch_frames, detections_batch):
                row = _row_from_detections(
                    detections,
                    scorer=scorer,
                    source_view=spec.view,
                    video_path=spec.path,
                    frame_index=frame_index,
                    sample_index=sample_index,
                    time_sec=time_sec,
                    sample_fps=sample_fps,
                    source_fps=spec.fps,
                    frame_width=spec.width,
                    frame_height=spec.height,
                    frame=frame,
                )
                row["stage"] = stage
                row["evidence_score"] = round(_row_evidence_score(row), 6)
                rows.append(row)
            timing.postprocess_sec += time.perf_counter() - post_start
            batch_frames.clear()
            batch_meta.clear()

        for local_index, target_sec in enumerate(times):
            decode_start = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, target_sec) * 1000.0)
            ok, frame = cap.read()
            timing.decode_sec += time.perf_counter() - decode_start
            if not ok or frame is None:
                continue
            frame_index = int(round(target_sec * spec.fps))
            batch_frames.append(frame)
            batch_meta.append((frame_index, sample_index_base + local_index, target_sec))
            if len(batch_frames) >= batch_size:
                flush()
        flush()
    finally:
        cap.release()
    timing.sampled_frames = len(rows)
    timing.wall_sec = round(time.perf_counter() - wall_start, 6)
    timing.decode_sec = round(timing.decode_sec, 6)
    timing.inference_sec = round(timing.inference_sec, 6)
    timing.postprocess_sec = round(timing.postprocess_sec, 6)
    if timing.wall_sec > 0:
        timing.effective_fps = round(timing.sampled_frames / timing.wall_sec, 3)
    return rows, timing


def _cluster_episode_rows(
    rows: list[dict[str, Any]],
    *,
    min_score: float,
    max_gap_sec: float,
    pre_roll_sec: float,
    post_roll_sec: float,
    min_duration_sec: float,
    duration_sec: float,
) -> list[dict[str, Any]]:
    evidence_rows = [
        row
        for row in sorted(rows, key=lambda item: float(item.get("time_sec", 0.0) or 0.0))
        if float(row.get("evidence_score", 0.0) or 0.0) >= min_score
    ]
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_t: float | None = None
    for row in evidence_rows:
        t = float(row.get("time_sec", 0.0) or 0.0)
        if current and last_t is not None and t - last_t > max_gap_sec:
            clusters.append(current)
            current = []
        current.append(row)
        last_t = t
    if current:
        clusters.append(current)

    episodes: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        start = max(0.0, min(float(row.get("time_sec", 0.0) or 0.0) for row in cluster) - pre_roll_sec)
        end = min(duration_sec, max(float(row.get("time_sec", 0.0) or 0.0) for row in cluster) + post_roll_sec)
        if end - start < min_duration_sec:
            pad = (min_duration_sec - (end - start)) / 2.0
            start = max(0.0, start - pad)
            end = min(duration_sec, end + pad)
        labels = Counter()
        interactions = 0
        scores = []
        for row in cluster:
            labels.update(_labels(row))
            interactions += len(row.get("hand_object_interactions") or [])
            scores.append(float(row.get("evidence_score", 0.0) or 0.0))
        episodes.append(
            {
                "episode_id": f"episode_{index:06d}",
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "support_frame_count": len(cluster),
                "interaction_count": interactions,
                "confidence": round(max(scores) if scores else 0.0, 6),
                "support_labels": dict(labels.most_common()),
            }
        )
    return episodes


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter()
    interaction_count = 0
    strong_rows = 0
    for row in rows:
        labels.update(_labels(row))
        interaction_count += len(row.get("hand_object_interactions") or [])
        if float(row.get("evidence_score", 0.0) or 0.0) >= 0.65:
            strong_rows += 1
    return {
        "row_count": len(rows),
        "strong_evidence_rows": strong_rows,
        "interaction_count": interaction_count,
        "label_counts": dict(labels.most_common()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir or (Path("outputs") / f"yolo_only_probe_{_now_id()}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        _probe_video(Path(args.cam01) / "rgb.mp4", "cam01_third_person", Path(args.third_person_model)),
        _probe_video(Path(args.cam02) / "rgb.mp4", "cam02_first_person", Path(args.first_person_model)),
    ]

    total_start = time.perf_counter()
    model_load_rows = []
    models: dict[str, Any] = {}
    for spec in specs:
        load_start = time.perf_counter()
        models[spec.view] = _load_yolo_model(None, spec.model_path)
        model_load_rows.append(
            {
                "view": spec.view,
                "model_path": str(spec.model_path),
                "model_load_sec": round(time.perf_counter() - load_start, 6),
            }
        )

    all_timing: list[TimingRow] = []
    coarse_by_view: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        view_rows: list[dict[str, Any]] = []
        scorer = YoloActivityScorer(active_threshold=args.active_threshold, continuity_frames=args.continuity_frames)
        for idx, (start, end) in enumerate(_ranges(spec.duration_sec, args.chunk_sec), start=1):
            rows, timing = _scan_window(
                spec=spec,
                model=models[spec.view],
                stage="coarse_yolo_scan",
                start_sec=start,
                end_sec=end,
                sample_fps=args.coarse_fps,
                batch_size=args.batch_size,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                imgsz=args.imgsz,
                scorer=scorer,
                chunk_id=f"{spec.view}_chunk_{idx:03d}",
            )
            view_rows.extend(rows)
            all_timing.append(timing)
            print(
                f"[coarse] {spec.view} chunk {idx:03d}: frames={timing.sampled_frames} "
                f"wall={timing.wall_sec:.2f}s infer={timing.inference_sec:.2f}s"
            )
        coarse_by_view[spec.view] = view_rows
        _write_jsonl(output_dir / f"{spec.view}_coarse_rows.jsonl", view_rows)

    main_rows = coarse_by_view["cam01_third_person"]
    cluster_start = time.perf_counter()
    episodes = _cluster_episode_rows(
        main_rows,
        min_score=args.episode_min_score,
        max_gap_sec=args.episode_max_gap_sec,
        pre_roll_sec=args.episode_pre_roll_sec,
        post_roll_sec=args.episode_post_roll_sec,
        min_duration_sec=args.episode_min_duration_sec,
        duration_sec=specs[0].duration_sec,
    )
    episode_cluster_sec = round(time.perf_counter() - cluster_start, 6)

    dense_rows: list[dict[str, Any]] = []
    for episode in episodes[: args.max_episodes]:
        scorer = YoloActivityScorer(active_threshold=args.active_threshold, continuity_frames=args.continuity_frames)
        rows, timing = _scan_window(
            spec=specs[0],
            model=models["cam01_third_person"],
            stage="dense_episode_scan",
            start_sec=float(episode["start_sec"]),
            end_sec=float(episode["end_sec"]),
            sample_fps=args.dense_fps,
            batch_size=args.batch_size,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            imgsz=args.imgsz,
            scorer=scorer,
            chunk_id=str(episode["episode_id"]),
        )
        dense_rows.extend(rows)
        all_timing.append(timing)
        print(
            f"[dense] {episode['episode_id']}: {episode['start_sec']:.1f}-{episode['end_sec']:.1f}s "
            f"frames={timing.sampled_frames} wall={timing.wall_sec:.2f}s infer={timing.inference_sec:.2f}s"
        )

    _write_jsonl(output_dir / "dense_episode_rows.jsonl", dense_rows)
    _write_jsonl(output_dir / "chunk_timing.jsonl", [asdict(row) for row in all_timing])
    _write_json(output_dir / "experiment_episodes.json", episodes)

    stage_totals: dict[str, dict[str, float]] = {}
    for row in all_timing:
        total = stage_totals.setdefault(
            row.stage,
            {
                "wall_sec": 0.0,
                "decode_sec": 0.0,
                "inference_sec": 0.0,
                "postprocess_sec": 0.0,
                "sampled_frames": 0.0,
            },
        )
        total["wall_sec"] += row.wall_sec
        total["decode_sec"] += row.decode_sec
        total["inference_sec"] += row.inference_sec
        total["postprocess_sec"] += row.postprocess_sec
        total["sampled_frames"] += row.sampled_frames
    for value in stage_totals.values():
        for key in list(value):
            value[key] = round(value[key], 6)
        value["effective_fps"] = round(value["sampled_frames"] / value["wall_sec"], 3) if value["wall_sec"] else 0.0

    summary = {
        "schema_version": "yolo_only_long_video_probe.v1",
        "output_dir": str(output_dir),
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "videos": [asdict(spec) for spec in specs],
        "model_load": model_load_rows,
        "config": {
            "coarse_fps": args.coarse_fps,
            "dense_fps": args.dense_fps,
            "chunk_sec": args.chunk_sec,
            "batch_size": args.batch_size,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "vlm_enabled": False,
        },
        "coarse_summary": {view: _summarize_rows(rows) for view, rows in coarse_by_view.items()},
        "episode_cluster_sec": episode_cluster_sec,
        "episode_count": len(episodes),
        "episodes": episodes,
        "dense_summary": _summarize_rows(dense_rows),
        "stage_totals": stage_totals,
        "total_wall_sec": round(time.perf_counter() - total_start, 6),
        "artifacts": {
            "chunk_timing": str(output_dir / "chunk_timing.jsonl"),
            "episodes": str(output_dir / "experiment_episodes.json"),
            "dense_rows": str(output_dir / "dense_episode_rows.jsonl"),
            "cam01_coarse_rows": str(output_dir / "cam01_third_person_coarse_rows.jsonl"),
            "cam02_coarse_rows": str(output_dir / "cam02_first_person_coarse_rows.jsonl"),
        },
    }
    _write_json(output_dir / "performance_timing.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO-only long-video coarse/dense timing probe.")
    parser.add_argument("--cam01", default=r"C:\Users\Xx7\Desktop\cam01")
    parser.add_argument("--cam02", default=r"C:\Users\Xx7\Desktop\cam02")
    parser.add_argument("--third-person-model", default=r"D:\LabCapability\LabSOPGuard\models\yolo\third_person\best.pt")
    parser.add_argument("--first-person-model", default=r"D:\LabCapability\LabSOPGuard\models\yolo\first_person\best.pt")
    parser.add_argument("--output-dir")
    parser.add_argument("--coarse-fps", type=float, default=0.25)
    parser.add_argument("--dense-fps", type=float, default=6.0)
    parser.add_argument("--chunk-sec", type=float, default=600.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--active-threshold", type=float, default=0.18)
    parser.add_argument("--continuity-frames", type=int, default=3)
    parser.add_argument("--episode-min-score", type=float, default=0.65)
    parser.add_argument("--episode-max-gap-sec", type=float, default=120.0)
    parser.add_argument("--episode-pre-roll-sec", type=float, default=20.0)
    parser.add_argument("--episode-post-roll-sec", type=float, default=20.0)
    parser.add_argument("--episode-min-duration-sec", type=float, default=30.0)
    parser.add_argument("--max-episodes", type=int, default=6)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

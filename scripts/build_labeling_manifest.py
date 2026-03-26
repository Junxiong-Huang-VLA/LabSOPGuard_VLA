from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frame-level labeling manifest for manual annotation.")
    parser.add_argument("--video-manifest", default="data/interim/video_manifest.csv")
    parser.add_argument("--out-dir", default="data/interim/labeling")
    parser.add_argument("--sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--max-frames-per-video", type=int, default=120)
    return parser.parse_args()


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _extract_sample_frames(
    video_path: Path,
    out_dir: Path,
    sample_id: str,
    interval_sec: float,
    max_frames: int,
) -> List[Dict[str, str]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    step = max(1, int(round(interval_sec * fps)))

    rows: List[Dict[str, str]] = []
    frame_id = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % step != 0:
            frame_id += 1
            continue

        frame_name = f"{sample_id}__f{frame_id:06d}.jpg"
        frame_path = out_dir / "frames" / sample_id / frame_name
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        if not _safe_write_jpg(frame_path, frame):
            frame_id += 1
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "video_path": str(video_path),
                "frame_id": str(frame_id),
                "timestamp_sec": f"{frame_id / fps:.3f}",
                "frame_path": str(frame_path).replace("\\", "/"),
                "annotation_status": "pending",
                "labels_json": "[]",
            }
        )
        kept += 1
        if kept >= max_frames:
            break
        frame_id += 1
    cap.release()
    return rows


def _safe_write_jpg(path: Path, frame) -> bool:
    # cv2.imwrite may fail on Windows for unicode paths; fallback to imencode+tofile.
    ok = cv2.imwrite(str(path), frame)
    if ok:
        return True
    ok2, buf = cv2.imencode(".jpg", frame)
    if not ok2:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.video_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"video manifest not found: {manifest_path}")

    rows = _read_manifest(manifest_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, str]] = []
    for row in rows:
        if str(row.get("valid_status", row.get("pair_status", ""))).lower() not in {"valid", "paired"}:
            continue
        sample_id = row.get("sample_id", "unknown")
        video = row.get("rgb_path", "")
        if not video:
            continue
        out_rows.extend(
            _extract_sample_frames(
                video_path=Path(video),
                out_dir=out_dir,
                sample_id=sample_id,
                interval_sec=args.sample_interval_sec,
                max_frames=args.max_frames_per_video,
            )
        )

    manifest_csv = out_dir / "labeling_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "sample_id",
            "video_path",
            "frame_id",
            "timestamp_sec",
            "frame_path",
            "annotation_status",
            "labels_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    summary_json = out_dir / "labeling_manifest_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "source_manifest": str(manifest_path).replace("\\", "/"),
                "total_frames_for_labeling": len(out_rows),
                "output_manifest_csv": str(manifest_csv).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"labeling manifest: {manifest_csv}")
    print(f"summary: {summary_json}")
    print(f"frames for labeling: {len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

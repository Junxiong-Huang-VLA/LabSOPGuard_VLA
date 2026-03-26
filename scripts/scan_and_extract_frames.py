from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}


@dataclass
class VideoPair:
    sample_id: str
    rgb_path: Optional[Path]
    depth_path: Optional[Path]
    pair_status: str
    rgb_readable: bool
    depth_readable: bool
    rgb_frames: int
    depth_frames: int


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("scan_extract")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def scan_video_pairs(
    dataset_root: Path,
    rgb_suffix: str,
    depth_suffix: str,
    recursive: bool,
) -> List[VideoPair]:
    rgb_pat = re.compile(rf"(.+?){re.escape(rgb_suffix)}$", re.IGNORECASE)
    depth_pat = re.compile(rf"(.+?){re.escape(depth_suffix)}$", re.IGNORECASE)

    files = dataset_root.rglob("*") if recursive else dataset_root.iterdir()
    rgb_map: Dict[str, Path] = {}
    depth_map: Dict[str, Path] = {}

    for path in files:
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        stem = path.stem
        rgb_match = rgb_pat.match(stem)
        depth_match = depth_pat.match(stem)
        if rgb_match:
            rgb_map[rgb_match.group(1)] = path
        if depth_match:
            depth_map[depth_match.group(1)] = path

    pairs: List[VideoPair] = []
    for sample_id in sorted(set(rgb_map) | set(depth_map)):
        rgb_path = rgb_map.get(sample_id)
        depth_path = depth_map.get(sample_id)
        status = "paired" if rgb_path and depth_path else "missing_pair"
        rgb_readable, rgb_frames = probe_video(rgb_path)
        depth_readable, depth_frames = probe_video(depth_path)
        pairs.append(
            VideoPair(
                sample_id=sample_id,
                rgb_path=rgb_path,
                depth_path=depth_path,
                pair_status=status,
                rgb_readable=rgb_readable,
                depth_readable=depth_readable,
                rgb_frames=rgb_frames,
                depth_frames=depth_frames,
            )
        )
    return pairs


def probe_video(path: Optional[Path]) -> tuple[bool, int]:
    if path is None:
        return False, -1
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False, -1
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (frames > 0), frames


def write_manifest_csv(path: Path, pairs: List[VideoPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "rgb_path",
                "depth_path",
                "has_rgb",
                "has_depth",
                "pair_status",
                "rgb_readable",
                "depth_readable",
                "rgb_frames",
                "depth_frames",
                "valid_status",
            ]
        )
        for item in pairs:
            valid_status = (
                "valid"
                if item.pair_status == "paired" and item.rgb_readable and item.depth_readable
                else "invalid"
            )
            writer.writerow(
                [
                    item.sample_id,
                    str(item.rgb_path) if item.rgb_path else "",
                    str(item.depth_path) if item.depth_path else "",
                    int(item.rgb_path is not None),
                    int(item.depth_path is not None),
                    item.pair_status,
                    int(item.rgb_readable),
                    int(item.depth_readable),
                    item.rgb_frames,
                    item.depth_frames,
                    valid_status,
                ]
            )


def extract_frames(
    video_path: Path,
    output_dir: Path,
    sample_id: str,
    modality: str,
    interval_sec: float,
    max_frames: int,
    jpg_quality: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0
    step = max(1, int(round(interval_sec * fps)))

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            out_name = f"{sample_id}__{modality}__f{frame_idx:06d}.jpg"
            out_path = output_dir / out_name
            if _safe_imwrite_jpg(out_path, frame, jpg_quality):
                saved += 1
                if 0 < max_frames <= saved:
                    break
        frame_idx += 1
    cap.release()
    return saved


def _safe_imwrite_jpg(path: Path, image, jpg_quality: int) -> bool:
    # cv2.imwrite may fail on Windows for unicode paths; fallback to imencode+tofile.
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    if ok:
        return True
    encode_ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    if not encode_ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def write_report(
    report_path: Path,
    dataset_root: Path,
    pairs: List[VideoPair],
    extracted: Dict[str, Dict[str, int]],
) -> None:
    paired = sum(1 for p in pairs if p.pair_status == "paired")
    valid = sum(1 for p in pairs if p.pair_status == "paired" and p.rgb_readable and p.depth_readable)
    report = {
        "dataset_root": str(dataset_root),
        "total_samples": len(pairs),
        "paired_samples": paired,
        "missing_pair_samples": len(pairs) - paired,
        "valid_samples": valid,
        "invalid_samples": len(pairs) - valid,
        "extracted_frames": extracted,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan valid RGB/Depth video pairs and extract frames."
    )
    parser.add_argument("--dataset-root", default="D:\\labdata")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--rgb-suffix", default="_rgb")
    parser.add_argument("--depth-suffix", default="_depth")
    parser.add_argument("--manifest-csv", default="data/interim/video_manifest.csv")
    parser.add_argument("--report-json", default="outputs/reports/video_scan_report.json")
    parser.add_argument("--frames-root", default="data/interim/frames")
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--max-frames-per-video", type=int, default=60)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--jpg-quality", type=int, default=90)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.verbose)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    pairs = scan_video_pairs(
        dataset_root=dataset_root,
        rgb_suffix=args.rgb_suffix,
        depth_suffix=args.depth_suffix,
        recursive=args.recursive,
    )
    write_manifest_csv(Path(args.manifest_csv), pairs)

    extracted: Dict[str, Dict[str, int]] = {}
    if not args.skip_extract:
        paired_items = [p for p in pairs if p.pair_status == "paired" and p.rgb_readable and p.depth_readable]
        if args.max_videos > 0:
            paired_items = paired_items[: args.max_videos]

        frames_root = Path(args.frames_root)
        for item in paired_items:
            extracted[item.sample_id] = {}
            if item.rgb_path is not None:
                rgb_count = extract_frames(
                    video_path=item.rgb_path,
                    output_dir=frames_root / item.sample_id / "rgb",
                    sample_id=item.sample_id,
                    modality="rgb",
                    interval_sec=args.interval_sec,
                    max_frames=args.max_frames_per_video,
                    jpg_quality=args.jpg_quality,
                )
                extracted[item.sample_id]["rgb"] = rgb_count
            if not args.skip_depth and item.depth_path is not None:
                depth_count = extract_frames(
                    video_path=item.depth_path,
                    output_dir=frames_root / item.sample_id / "depth",
                    sample_id=item.sample_id,
                    modality="depth",
                    interval_sec=args.interval_sec,
                    max_frames=args.max_frames_per_video,
                    jpg_quality=args.jpg_quality,
                )
                extracted[item.sample_id]["depth"] = depth_count

    write_report(Path(args.report_json), dataset_root, pairs, extracted)

    paired = sum(1 for p in pairs if p.pair_status == "paired")
    valid = sum(1 for p in pairs if p.pair_status == "paired" and p.rgb_readable and p.depth_readable)
    logger.info("total samples: %d", len(pairs))
    logger.info("paired samples: %d", paired)
    logger.info("missing pair: %d", len(pairs) - paired)
    logger.info("valid paired samples: %d", valid)
    logger.info("invalid samples: %d", len(pairs) - valid)
    logger.info("manifest: %s", args.manifest_csv)
    logger.info("report: %s", args.report_json)
    if not args.skip_extract:
        logger.info("frames root: %s", args.frames_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import read_jsonl, WorkbenchROI
from .video_utils import default_roi


def _roi_tuple(width: int, height: int, roi: WorkbenchROI | dict[str, Any] | None) -> tuple[int, int, int, int]:
    if roi is None:
        return default_roi(width, height)
    if isinstance(roi, WorkbenchROI):
        x, y, w, h = roi.x, roi.y, roi.w, roi.h
    else:
        x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
    x = max(0, min(int(x), max(0, width - 1)))
    y = max(0, min(int(y), max(0, height - 1)))
    w = max(1, min(int(w), max(1, width - x)))
    h = max(1, min(int(h), max(1, height - y)))
    return x, y, w, h


def _placeholder(width: int = 1280, height: int = 720, text: str = "dry-run preview") -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for debug visualization") from exc

    image = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.putText(image, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 40), 2, cv2.LINE_AA)
    return image


def save_roi_preview(
    video_path: str | Path,
    roi: WorkbenchROI | dict[str, Any] | None,
    output_path: str | Path,
    dry_run: bool = False,
) -> Path:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for ROI preview") from exc

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = None
    source = Path(video_path)
    if source.exists() and not dry_run:
        cap = cv2.VideoCapture(str(source))
        try:
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if frame_count > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count - 1, max(0, frame_count // 10)))
                ok, candidate = cap.read()
                if ok:
                    frame = candidate
        finally:
            cap.release()
    if frame is None:
        frame = _placeholder(text="ROI preview placeholder")

    height, width = frame.shape[:2]
    x, y, w, h = _roi_tuple(width, height, roi)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 3)
    cv2.putText(frame, f"ROI x={x} y={y} w={w} h={h}", (x, max(24, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 220), 2, cv2.LINE_AA)
    cv2.imwrite(str(target), frame)
    return target


def save_frame_score_plot(
    frame_scores_path: str | Path,
    detected_segments_path: str | Path,
    output_path: str | Path,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for frame score plots") from exc

    frame_scores = read_jsonl(frame_scores_path) if Path(frame_scores_path).exists() else []
    segments = read_jsonl(detected_segments_path) if Path(detected_segments_path).exists() else []
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    xs = [float(row.get("local_time_sec", row.get("time_sec", 0.0))) for row in frame_scores]
    motion = [float(row.get("motion_score", 0.0)) for row in frame_scores]
    active = [float(row.get("active_score", 0.0)) for row in frame_scores]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    if xs:
        ax.plot(xs, motion, label="motion_score", linewidth=1.2)
        ax.plot(xs, active, label="active_score", linewidth=1.2, alpha=0.8)
    else:
        ax.text(0.5, 0.5, "No frame scores", ha="center", va="center", transform=ax.transAxes)

    for segment in segments:
        start = float(segment.get("start_sec", 0.0))
        end = float(segment.get("end_sec", start))
        ax.axvspan(start, end, color="#ffcc66", alpha=0.28)
        ax.axvline(start, color="#cc7a00", linewidth=0.7)
        ax.axvline(end, color="#cc7a00", linewidth=0.7)

    ax.set_xlabel("local time (sec)")
    ax.set_ylabel("score")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return target


def _read_or_placeholder(path: Path, width: int = 300, height: int = 180, label: str = "") -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for contact sheet generation") from exc

    image = cv2.imread(str(path)) if path.exists() else None
    if image is None:
        image = np.full((height, width, 3), 230, dtype=np.uint8)
        cv2.putText(image, "missing frame", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
        if label:
            cv2.putText(image, label, (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    return cv2.resize(image, (width, height))


def save_segment_contact_sheet(
    keyframes_dir: str | Path,
    detected_segments_path: str | Path,
    output_path: str | Path,
) -> Path:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for contact sheet generation") from exc

    segments = read_jsonl(detected_segments_path) if Path(detected_segments_path).exists() else []
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 300, 180
    header_h = 58
    margin = 14
    if not segments:
        image = _placeholder(960, 240, "No detected segments")
        cv2.imwrite(str(target), image)
        return target

    rows: list[np.ndarray] = []
    root = Path(keyframes_dir)
    for segment in segments:
        segment_id = str(segment.get("segment_id", "segment"))
        frames = [
            root / segment_id / "third_person_start.jpg",
            root / segment_id / "third_person_middle.jpg",
            root / segment_id / "third_person_end.jpg",
        ]
        row_w = tile_w * 3 + margin * 4
        row_h = header_h + tile_h + margin
        row = np.full((row_h, row_w, 3), 248, dtype=np.uint8)
        label = (
            f"{segment_id}  {float(segment.get('start_sec', 0.0)):.1f}s-"
            f"{float(segment.get('end_sec', 0.0)):.1f}s  "
            f"dur={float(segment.get('duration_sec', 0.0)):.1f}s  "
            f"active={float(segment.get('avg_active_score', 0.0)):.2f}"
        )
        cv2.putText(row, label, (margin, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
        for idx, frame_path in enumerate(frames):
            frame = _read_or_placeholder(frame_path, tile_w, tile_h, frame_path.name)
            x = margin + idx * (tile_w + margin)
            y = header_h
            row[y : y + tile_h, x : x + tile_w] = frame
        rows.append(row)

    sheet_h = sum(row.shape[0] for row in rows) + margin * max(0, len(rows) - 1)
    sheet_w = max(row.shape[1] for row in rows)
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)
    y = 0
    for row in rows:
        sheet[y : y + row.shape[0], 0 : row.shape[1]] = row
        y += row.shape[0] + margin
    max_height = 16000
    if sheet.shape[0] > max_height:
        scale = max_height / sheet.shape[0]
        sheet = cv2.resize(sheet, (max(1, math.floor(sheet.shape[1] * scale)), max_height))
    cv2.imwrite(str(target), sheet)
    return target

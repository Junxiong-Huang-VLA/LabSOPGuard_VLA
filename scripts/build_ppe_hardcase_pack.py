from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_summary_video_map(summary_path: Path) -> Dict[str, str]:
    data = _load_json(summary_path)
    if not isinstance(data, list):
        return {}
    out: Dict[str, str] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("sample_id", "")).strip()
        video = str(row.get("video", row.get("video_path", ""))).strip()
        if sid and video:
            out[sid] = video
    return out


def _safe_imwrite_jpg(path: Path, image, quality: int = 92) -> bool:
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if ok:
        return True
    enc_ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not enc_ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def _extract_frames(video_path: str, frame_ids: List[int]) -> Dict[int, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    uniq = sorted(set(int(x) for x in frame_ids if int(x) >= 0))
    frames: Dict[int, Any] = {}
    for fid in uniq:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fid))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames[fid] = frame
    cap.release()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PPE hardcase pack from violation diagnostics")
    parser.add_argument("--diagnostics-json", default="outputs/reports/violation_diagnostics.json")
    parser.add_argument("--batch-monitor-summary", default="outputs/predictions/batch_monitor/summary.json")
    parser.add_argument("--rule-id", default="missing_ppe")
    parser.add_argument("--context", type=int, default=3, help="context radius in frame units")
    parser.add_argument("--out-dir", default="data/interim/hardcases/ppe_missing")
    parser.add_argument("--out-csv", default="data/interim/hardcases/ppe_missing_manifest.csv")
    args = parser.parse_args()

    diagnostics = _load_json(Path(args.diagnostics_json))
    rows = diagnostics.get("diagnostics", []) if isinstance(diagnostics, dict) else []
    video_map = _read_summary_video_map(Path(args.batch_monitor_summary))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("rule_id", "")) != args.rule_id:
            continue
        sid = str(row.get("sample_id", "")).strip()
        if not sid:
            continue
        video = video_map.get(sid, "")
        if not video:
            continue

        center = int(row.get("frame_id", -1))
        if center < 0:
            continue

        frame_ids = [center + d for d in range(-args.context, args.context + 1) if center + d >= 0]
        frames = _extract_frames(video, frame_ids)
        sample_dir = out_dir / sid
        sample_dir.mkdir(parents=True, exist_ok=True)

        for fid in frame_ids:
            frame = frames.get(fid)
            if frame is None:
                continue
            rel = Path(sid) / f"{sid}__f{fid:06d}.jpg"
            dst = out_dir / rel
            if not _safe_imwrite_jpg(dst, frame):
                continue
            manifest_rows.append(
                {
                    "sample_id": sid,
                    "video_path": video,
                    "rule_id": row.get("rule_id", ""),
                    "center_frame_id": center,
                    "frame_id": fid,
                    "is_center": int(fid == center),
                    "missing_ppe_items": "|".join(row.get("missing_ppe_items", [])),
                    "image_path": str(dst).replace("\\", "/"),
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "video_path",
        "rule_id",
        "center_frame_id",
        "frame_id",
        "is_center",
        "missing_ppe_items",
        "image_path",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in manifest_rows:
            w.writerow({k: r.get(k) for k in fields})

    print(f"hardcase_dir: {out_dir}")
    print(f"manifest_csv: {out_csv}")
    print(f"frames: {len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

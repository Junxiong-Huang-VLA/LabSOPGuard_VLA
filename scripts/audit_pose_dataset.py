from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit YOLO pose dataset quality.")
    p.add_argument("--dataset-yaml", default="data/processed/yolo_pose_dataset/dataset.yaml")
    p.add_argument("--out-json", default="outputs/reports/pose_dataset_audit.json")
    return p.parse_args()


def _iter_label_files(root: Path, split: str) -> List[Path]:
    d = root / "labels" / split
    if not d.exists():
        return []
    return sorted(d.glob("*.txt"))


def main() -> int:
    args = parse_args()
    ds_path = Path(args.dataset_yaml)
    if not ds_path.exists():
        raise FileNotFoundError(f"dataset yaml not found: {ds_path}")
    ds = yaml.safe_load(ds_path.read_text(encoding="utf-8")) or {}
    root = Path(ds["path"])
    names: Dict[int, str] = {
        int(k): str(v) for k, v in (ds.get("names", {}) or {}).items()
    }

    cls_counter: Counter = Counter()
    token_len: Counter = Counter()
    vis_counter: Counter = Counter()
    quad_counter: Counter = Counter()
    box_w: List[float] = []
    box_h: List[float] = []
    line_count = 0

    for split in ("train", "val"):
        for f in _iter_label_files(root, split):
            for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = ln.strip()
                if not s:
                    continue
                t = s.split()
                if len(t) < 5:
                    continue
                line_count += 1
                token_len[len(t)] += 1
                cls = int(float(t[0]))
                cls_counter[cls] += 1
                x, y, w, h = map(float, t[1:5])
                box_w.append(w)
                box_h.append(h)
                quad = ("L" if x < 0.5 else "R") + ("T" if y < 0.5 else "B")
                quad_counter[quad] += 1
                if len(t) >= 14:
                    for v in t[7::3]:
                        vis_counter[int(float(v))] += 1

    report = {
        "dataset_yaml": str(ds_path).replace("\\", "/"),
        "root": str(root).replace("\\", "/"),
        "lines": line_count,
        "class_distribution": {names.get(k, str(k)): v for k, v in sorted(cls_counter.items())},
        "class_ids_present": sorted(cls_counter.keys()),
        "token_length_distribution": {str(k): v for k, v in sorted(token_len.items())},
        "keypoint_visibility_distribution": {str(k): v for k, v in sorted(vis_counter.items())},
        "box_stats": {
            "width_mean": (sum(box_w) / len(box_w)) if box_w else 0.0,
            "height_mean": (sum(box_h) / len(box_h)) if box_h else 0.0,
            "width_min": min(box_w) if box_w else 0.0,
            "height_min": min(box_h) if box_h else 0.0,
        },
        "spatial_quadrant_distribution": dict(quad_counter),
        "warnings": [],
    }

    if report["class_ids_present"] and max(report["class_ids_present"]) <= 4:
        report["warnings"].append("Only class IDs 0-4 are present; higher classes are absent.")
    if vis_counter and vis_counter.get(0, 0) > vis_counter.get(1, 0) + vis_counter.get(2, 0):
        report["warnings"].append("Most keypoints are invisible (v=0), pose supervision is weak.")
    if report["box_stats"]["width_mean"] < 0.05:
        report["warnings"].append("Average bbox width is very small; may overfit to tiny regions.")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


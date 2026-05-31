from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


TARGET_FIELDS = [
    "sample_id",
    "target_name",
    "region_reference",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "center_x",
    "center_y",
    "center_depth",
    "region_depth_mean",
    "region_depth_median",
    "valid_depth_ratio",
    "xyz_x",
    "xyz_y",
    "xyz_z",
]


def build_target_representation(
    sample_id: str,
    target_name: str,
    bbox: List[int],
    center_point: List[float],
    center_depth: float,
    region_depth_mean: float,
    region_depth_median: float,
    valid_depth_ratio: float,
    region_reference: str,
    xyz: List[float] | None,
    flags: List[str] | None = None,
) -> Dict[str, Any]:
    rep = {
        "sample_id": sample_id,
        "target_name": target_name,
        "region_reference": region_reference,
        "bbox": bbox,
        "center_point": [float(center_point[0]), float(center_point[1])],
        "center_depth": float(center_depth),
        "region_depth_mean": float(region_depth_mean),
        "region_depth_median": float(region_depth_median),
        "valid_depth_ratio": float(valid_depth_ratio),
        "xyz": xyz,
        "quality_flags": flags or [],
        "action_input": {
            "target_center_uvd": [float(center_point[0]), float(center_point[1]), float(center_depth)],
            "target_xyz": xyz,
            "depth_confidence": float(valid_depth_ratio),
        },
    }
    return rep


def to_flat_row(rep: Dict[str, Any]) -> Dict[str, Any]:
    bbox = rep.get("bbox") or [0, 0, 0, 0]
    center = rep.get("center_point") or [0.0, 0.0]
    xyz = rep.get("xyz") or [None, None, None]
    return {
        "sample_id": rep.get("sample_id"),
        "target_name": rep.get("target_name"),
        "region_reference": rep.get("region_reference"),
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "center_x": center[0],
        "center_y": center[1],
        "center_depth": rep.get("center_depth"),
        "region_depth_mean": rep.get("region_depth_mean"),
        "region_depth_median": rep.get("region_depth_median"),
        "valid_depth_ratio": rep.get("valid_depth_ratio"),
        "xyz_x": xyz[0],
        "xyz_y": xyz[1],
        "xyz_z": xyz[2],
    }


def export_target_json(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, indent=2)


def export_target_csv(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(to_flat_row(row))

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO pose dataset from existing YOLO detection labels."
    )
    parser.add_argument(
        "--src-root",
        default="data/processed/yolo_dataset",
        help="Source detection dataset root with images/{train,val} labels/{train,val}.",
    )
    parser.add_argument(
        "--out-root",
        default="data/processed/yolo_pose_dataset",
        help="Output pose dataset root.",
    )
    parser.add_argument(
        "--schema",
        default="configs/data/pose_keypoints_schema.yaml",
        help="Pose keypoint schema yaml.",
    )
    parser.add_argument(
        "--class-yaml",
        default=None,
        help="Optional dataset yaml providing class names. If omitted, use <src-root>/dataset.yaml.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of hard-linking.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean out-root before rebuild.",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _safe_link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _load_class_names(src_yaml: Path) -> Dict[int, str]:
    raw = _read_yaml(src_yaml)
    names = raw.get("names", {})
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def _class_to_kp_count(class_name: str, schema: Dict) -> int:
    keypoint_names = schema.get("keypoint_names", {})
    cls_cfg = schema.get("class_alias", {})
    if class_name in keypoint_names:
        return len(keypoint_names[class_name])
    for canonical, aliases in cls_cfg.items():
        if class_name == canonical or class_name in aliases:
            if canonical in keypoint_names:
                return len(keypoint_names[canonical])
    return int(schema.get("default_keypoints", 3))


def _bbox_from_polygon(poly_xy: List[float]) -> Tuple[float, float, float, float]:
    xs = poly_xy[0::2]
    ys = poly_xy[1::2]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(1.0, max(xs))
    y2 = min(1.0, max(ys))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return cx, cy, w, h


def _canon_type(class_name: str, schema: Dict) -> str:
    if class_name in schema.get("keypoint_names", {}):
        return class_name
    for canon, aliases in (schema.get("class_alias", {}) or {}).items():
        if class_name == canon or class_name in aliases:
            return str(canon)
    return "default"


def _vessel_kps_from_poly(poly_xy: List[float]) -> List[Tuple[float, float, int]]:
    pts = [(poly_xy[i], poly_xy[i + 1]) for i in range(0, len(poly_xy), 2)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    left = min(pts, key=lambda p: p[0])
    right = max(pts, key=lambda p: p[0])
    return [(cx, cy, 2), (left[0], left[1], 2), (right[0], right[1], 2)]


def _tool_kps_from_poly(poly_xy: List[float]) -> List[Tuple[float, float, int]]:
    pts = [(poly_xy[i], poly_xy[i + 1]) for i in range(0, len(poly_xy), 2)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    dists = [((p[0] - cx) ** 2 + (p[1] - cy) ** 2, i) for i, p in enumerate(pts)]
    dists.sort(reverse=True)
    tip = pts[dists[0][1]]
    vx, vy = tip[0] - cx, tip[1] - cy
    norm = math.sqrt(vx * vx + vy * vy) + 1e-9
    vx, vy = vx / norm, vy / norm
    grasp = min(pts, key=lambda p: (p[0] - cx) * vx + (p[1] - cy) * vy)
    return [(tip[0], tip[1], 2), (cx, cy, 2), (grasp[0], grasp[1], 2)]


def _kps_from_bbox(
    cls_type: str,
    cx: float,
    cy: float,
    w: float,
    h: float,
    kp_count: int,
) -> List[Tuple[float, float, int]]:
    x1 = cx - w / 2.0
    x2 = cx + w / 2.0
    y1 = cy - h / 2.0
    y2 = cy + h / 2.0
    if cls_type == "target_vessel":
        kps = [(cx, cy, 1), (x1, cy, 1), (x2, cy, 1)]
    elif cls_type == "titration_tool":
        kps = [(cx, y1, 1), (cx, cy, 1), (cx, y2, 1)]
    else:
        kps = [(0.0, 0.0, 0)] * kp_count
    if len(kps) < kp_count:
        kps.extend([(0.0, 0.0, 0)] * (kp_count - len(kps)))
    return kps[:kp_count]


def _convert_label_line_to_pose(
    line: str,
    class_names: Dict[int, str],
    schema: Dict,
) -> Tuple[str, bool, bool]:
    parts = line.strip().split()
    if len(parts) < 5:
        return "", False, False

    cls_id = int(float(parts[0]))
    class_name = class_names.get(cls_id, str(cls_id))
    cls_type = _canon_type(class_name, schema)
    kp_count = _class_to_kp_count(class_name, schema)

    used_polygon = False
    if len(parts) > 5:
        coords = [float(v) for v in parts[1:]]
        if len(coords) >= 6 and len(coords) % 2 == 0:
            cx, cy, bw, bh = _bbox_from_polygon(coords)
            if cls_type == "target_vessel":
                kps = _vessel_kps_from_poly(coords)
            elif cls_type == "titration_tool":
                kps = _tool_kps_from_poly(coords)
            else:
                kps = _kps_from_bbox(cls_type, cx, cy, bw, bh, kp_count)
            used_polygon = True
        else:
            # fallback to first 4 numbers as bbox-like values
            cx, cy, bw, bh = [float(v) for v in parts[1:5]]
            kps = _kps_from_bbox(cls_type, cx, cy, bw, bh, kp_count)
    else:
        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
        kps = _kps_from_bbox(cls_type, cx, cy, bw, bh, kp_count)

    if len(kps) < kp_count:
        kps.extend([(0.0, 0.0, 0)] * (kp_count - len(kps)))
    kps = kps[:kp_count]

    kp_tokens: List[str] = []
    for x, y, v in kps:
        kp_tokens.extend([f"{x:.6f}", f"{y:.6f}", str(int(v))])
    out = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp_tokens)
    return out.strip(), True, used_polygon


def _iter_images(split_img_dir: Path) -> List[Path]:
    out: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        out.extend(split_img_dir.glob(ext))
    return sorted(out)


def main() -> int:
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    schema_path = Path(args.schema)
    src_yaml = Path(args.class_yaml) if args.class_yaml else (src_root / "dataset.yaml")

    if not src_root.exists():
        raise FileNotFoundError(f"src-root not found: {src_root}")
    if not src_yaml.exists():
        raise FileNotFoundError(f"source dataset.yaml not found: {src_yaml}")
    if not schema_path.exists():
        raise FileNotFoundError(f"schema not found: {schema_path}")

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    class_names = _load_class_names(src_yaml)
    schema = _read_yaml(schema_path)
    default_kp = int(schema.get("default_keypoints", 3))

    splits = ["train", "val"]
    summary: Dict[str, Dict[str, int]] = {}
    total_boxes = 0
    total_placeholder = 0
    total_poly_derived = 0

    for split in splits:
        src_img_dir = src_root / "images" / split
        src_lbl_dir = src_root / "labels" / split
        dst_img_dir = out_root / "images" / split
        dst_lbl_dir = out_root / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_count = 0
        label_count = 0
        box_count = 0

        for img_path in _iter_images(src_img_dir):
            image_count += 1
            dst_img = dst_img_dir / img_path.name
            _safe_link_or_copy(img_path, dst_img, copy_images=args.copy_images)

            src_lbl = src_lbl_dir / f"{img_path.stem}.txt"
            dst_lbl = dst_lbl_dir / f"{img_path.stem}.txt"
            if not src_lbl.exists():
                dst_lbl.write_text("", encoding="utf-8")
                continue

            out_lines: List[str] = []
            for line in src_lbl.read_text(encoding="utf-8").splitlines():
                pline, ok, used_poly = _convert_label_line_to_pose(line, class_names, schema)
                if ok:
                    out_lines.append(pline)
                    box_count += 1
                    total_placeholder += 1
                    if used_poly:
                        total_poly_derived += 1
            dst_lbl.write_text("\n".join(out_lines), encoding="utf-8")
            label_count += 1

        summary[split] = {
            "images": image_count,
            "label_files": label_count,
            "boxes": box_count,
        }
        total_boxes += box_count

    names = {idx: name for idx, name in sorted(class_names.items(), key=lambda x: x[0])}
    dataset_yaml = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": names,
        "kpt_shape": [default_kp, 3],
        "flip_idx": list(range(default_kp)),
    }
    (out_root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    report = {
        "src_root": str(src_root).replace("\\", "/"),
        "out_root": str(out_root).replace("\\", "/"),
        "note": "Bootstrapped pose labels with invisible placeholder keypoints (0,0,0). Replace with real keypoint annotations for meaningful pose training.",
        "splits": summary,
        "total_boxes": total_boxes,
        "pose_boxes_written": total_placeholder,
        "polygon_derived_keypoint_boxes": total_poly_derived,
        "bbox_heuristic_keypoint_boxes": total_placeholder - total_poly_derived,
        "schema": str(schema_path).replace("\\", "/"),
    }
    (out_root / "pose_build_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

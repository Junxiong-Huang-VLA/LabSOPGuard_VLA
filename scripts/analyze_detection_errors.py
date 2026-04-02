from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import yaml


@dataclass
class Box:
    cls_id: int
    xyxy: Tuple[float, float, float, float]
    conf: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-class detection error analysis on val set.")
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--out-json", default="outputs/reports/detection_error_report.json")
    parser.add_argument("--out-csv", default="outputs/reports/detection_error_report.csv")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all val images.")
    return parser.parse_args()


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def _xywhn_to_xyxy(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    bw = w * img_w
    bh = h * img_h
    cx = x * img_w
    cy = y * img_h
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return (x1, y1, x2, y2)


def _load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> List[Box]:
    if not label_path.exists():
        return []
    boxes: List[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        toks = s.split()
        if len(toks) < 5:
            continue
        try:
            cls_id = int(float(toks[0]))
            cx, cy, w, h = float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4])
        except ValueError:
            continue
        boxes.append(Box(cls_id=cls_id, xyxy=_xywhn_to_xyxy(cx, cy, w, h, img_w, img_h)))
    return boxes


def _group_by_class(boxes: Iterable[Box]) -> Dict[int, List[Box]]:
    out: Dict[int, List[Box]] = {}
    for b in boxes:
        out.setdefault(b.cls_id, []).append(b)
    return out


def _match_class(gt_boxes: List[Box], pred_boxes: List[Box], iou_thr: float) -> Tuple[int, int, int]:
    if not gt_boxes and not pred_boxes:
        return (0, 0, 0)
    if not gt_boxes:
        return (0, len(pred_boxes), 0)
    if not pred_boxes:
        return (0, 0, len(gt_boxes))

    matched_gt = set()
    matched_pred = set()
    candidates: List[Tuple[float, int, int]] = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou = _iou(pb.xyxy, gb.xyxy)
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True, key=lambda x: x[0])

    tp = 0
    for _, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        tp += 1
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    return (tp, fp, fn)


def main() -> int:
    args = parse_args()

    dataset_yaml = Path(args.dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {dataset_yaml}")
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    local_cfg = Path(".ultralytics")
    local_cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_cfg.resolve()))

    from ultralytics import YOLO  # type: ignore

    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    val_rel = Path(str(cfg["val"]).replace("\\", "/"))
    val_images_dir = val_rel if val_rel.is_absolute() else (root / val_rel)
    names_raw = cfg.get("names", {})
    names: Dict[int, str] = {int(k): str(v) for k, v in names_raw.items()}
    val_split = val_rel.parts[-1] if val_rel.parts else "val"
    labels_dir = root / "labels" / val_split
    if not labels_dir.exists() and (root / "labels" / "val").exists():
        labels_dir = root / "labels" / "val"
    image_paths = sorted([p for p in val_images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    model = YOLO(str(weights))

    per_class: Dict[int, Dict[str, int]] = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in names.keys()}
    fp_examples: Dict[int, List[str]] = {cid: [] for cid in names.keys()}
    fn_examples: Dict[int, List[str]] = {cid: [] for cid in names.keys()}

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = _load_gt_boxes(label_path, w, h)

        pred_result = model.predict(source=str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        pred_boxes: List[Box] = []
        if pred_result.boxes is not None and len(pred_result.boxes) > 0:
            for b in pred_result.boxes:
                xyxy = tuple(float(v) for v in b.xyxy[0].tolist())
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                pred_boxes.append(Box(cls_id=cls_id, xyxy=xyxy, conf=conf))

        gt_by_cls = _group_by_class(gt_boxes)
        pred_by_cls = _group_by_class(pred_boxes)
        cls_ids = set(gt_by_cls.keys()) | set(pred_by_cls.keys()) | set(names.keys())
        for cid in cls_ids:
            gt_c = gt_by_cls.get(cid, [])
            pred_c = pred_by_cls.get(cid, [])
            tp, fp, fn = _match_class(gt_c, pred_c, args.iou_thr)
            if cid not in per_class:
                per_class[cid] = {"tp": 0, "fp": 0, "fn": 0}
                fp_examples[cid] = []
                fn_examples[cid] = []
            per_class[cid]["tp"] += tp
            per_class[cid]["fp"] += fp
            per_class[cid]["fn"] += fn
            if fp > 0 and len(fp_examples[cid]) < 8:
                fp_examples[cid].append(img_path.name)
            if fn > 0 and len(fn_examples[cid]) < 8:
                fn_examples[cid].append(img_path.name)

    rows = []
    for cid in sorted(per_class.keys()):
        tp = per_class[cid]["tp"]
        fp = per_class[cid]["fp"]
        fn = per_class[cid]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "class_id": cid,
                "class_name": names.get(cid, str(cid)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "fp_examples": fp_examples.get(cid, []),
                "fn_examples": fn_examples.get(cid, []),
            }
        )

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_yaml": str(dataset_yaml),
        "weights": str(weights),
        "images_evaluated": len(image_paths),
        "conf": args.conf,
        "iou_thr": args.iou_thr,
        "per_class": rows,
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "class_name", "tp", "fp", "fn", "precision", "recall", "f1", "fp_examples", "fn_examples"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    **r,
                    "fp_examples": ";".join(r["fp_examples"]),
                    "fn_examples": ";".join(r["fn_examples"]),
                }
            )

    print(f"saved: {out_json}")
    print(f"saved: {out_csv}")
    print(f"images evaluated: {len(image_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

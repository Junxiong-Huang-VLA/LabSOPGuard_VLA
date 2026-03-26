from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert labeling manifest to YOLO detection dataset.")
    parser.add_argument("--manifest-csv", default="data/interim/labeling/labeling_manifest.csv")
    parser.add_argument("--class-schema", default="configs/data/class_schema.yaml")
    parser.add_argument("--out-dir", default="data/processed/yolo_dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--status-filter",
        default="done,verified",
        help="Only rows with annotation_status in this comma list will be converted.",
    )
    return parser.parse_args()


def _load_class_map(schema_path: Path) -> Tuple[Dict[str, int], List[str]]:
    cfg = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
    classes = cfg.get("classes", [])
    id_to_name = sorted([(int(c["id"]), str(c["name"])) for c in classes], key=lambda x: x[0])
    names = [name for _, name in id_to_name]
    name_to_id = {name: idx for idx, name in id_to_name}
    return name_to_id, names


def _bbox_xyxy_to_yolo(xyxy: List[float], w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = max(0.0, min(x1, w - 1))
    x2 = max(0.0, min(x2, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    y2 = max(0.0, min(y2, h - 1))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest_csv)
    schema_path = Path(args.class_schema)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"class schema not found: {schema_path}")

    name_to_id, class_names = _load_class_map(schema_path)
    valid_status = {s.strip().lower() for s in args.status_filter.split(",") if s.strip()}

    rows = _read_rows(manifest_path)
    usable = [r for r in rows if str(r.get("annotation_status", "")).lower() in valid_status]
    if not usable:
        print("No labeled rows found. Set annotation_status to done/verified and fill labels_json first.")
        return 0

    rng = random.Random(args.seed)
    rng.shuffle(usable)

    n = len(usable)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    splits = {
        "train": usable[:n_train],
        "val": usable[n_train:n_train + n_val],
        "test": usable[n_train + n_val:],
    }

    for split_name, split_rows in splits.items():
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in enumerate(split_rows):
            src_img = Path(row["frame_path"])
            if not src_img.is_absolute():
                src_img = Path.cwd() / src_img
            if not src_img.exists():
                continue

            img = cv2.imread(str(src_img))
            if img is None:
                continue
            h, w = img.shape[:2]

            sample_id = row.get("sample_id", "unknown")
            frame_id = row.get("frame_id", str(idx))
            out_stem = f"{sample_id}__f{int(float(frame_id)):06d}"
            dst_img = img_dir / f"{out_stem}.jpg"
            shutil.copy2(src_img, dst_img)

            labels = json.loads(row.get("labels_json", "[]") or "[]")
            yolo_lines: List[str] = []
            for ann in labels:
                class_name = str(ann.get("class_name", "")).strip()
                bbox = ann.get("bbox", [])
                if class_name not in name_to_id or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                xc, yc, bw, bh = _bbox_xyxy_to_yolo([float(v) for v in bbox], w, h)
                cls_id = name_to_id[class_name]
                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            (lbl_dir / f"{out_stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

    dataset_yaml = out_dir / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {i: n for i, n in enumerate(class_names)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "total_rows": len(rows),
        "usable_labeled_rows": len(usable),
        "splits": {k: len(v) for k, v in splits.items()},
        "dataset_yaml": str(dataset_yaml).replace("\\", "/"),
    }
    (out_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ImageRecord:
    src_img: Path
    label_lines: List[str]
    label_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standardized YOLO dataset from mixed frame/label layouts."
    )
    parser.add_argument("--src-root", default="data/interim/frames")
    parser.add_argument("--class-yaml", default="configs/data/class_schema.yaml")
    parser.add_argument("--out-root", default="data/dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--drop-invalid-label",
        action="store_true",
        help="Drop images whose label file exists but has invalid rows.",
    )
    return parser.parse_args()


def _line_is_valid(line: str) -> bool:
    t = line.strip().split()
    if len(t) < 5:
        return False
    try:
        _ = [float(x) for x in t]
    except Exception:
        return False
    return True


def _load_names(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = raw.get("names", {})
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, dict) and names:
        return {int(k): str(v) for k, v in names.items()}
    classes = raw.get("classes", [])
    if isinstance(classes, list):
        out: Dict[int, str] = {}
        for c in classes:
            if not isinstance(c, dict):
                continue
            if "id" not in c or "name" not in c:
                continue
            out[int(c["id"])] = str(c["name"])
        if out:
            return out
    return {}


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


def _iter_all_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _tail_key(name: str) -> str:
    s = name.lower()
    marker = "__f"
    idx = s.find(marker)
    return s[idx:] if idx >= 0 else s


def _build_coco_label_maps(src_root: Path, class_name_to_id: Dict[str, int]) -> Dict[str, List[str]]:
    ann_path = src_root / "train" / "_annotations.coco.json"
    if not ann_path.exists():
        return {}

    raw = json.loads(ann_path.read_text(encoding="utf-8"))
    images = raw.get("images", [])
    anns = raw.get("annotations", [])
    cats = raw.get("categories", [])

    image_meta: Dict[int, Dict] = {int(i["id"]): i for i in images if "id" in i}
    cat_id_to_name: Dict[int, str] = {int(c["id"]): str(c.get("name", c["id"])) for c in cats}
    cat_ids_sorted = sorted(cat_id_to_name.keys())
    cat_id_to_fallback = {cid: idx for idx, cid in enumerate(cat_ids_sorted)}

    by_file: Dict[str, List[str]] = {}
    by_tail: Dict[str, List[str]] = {}
    for a in anns:
        img_id = int(a.get("image_id", -1))
        cat_id = int(a.get("category_id", -1))
        bbox = a.get("bbox")
        meta = image_meta.get(img_id)
        if meta is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        w_img = float(meta.get("width", 0))
        h_img = float(meta.get("height", 0))
        if w_img <= 1 or h_img <= 1:
            continue

        name = cat_id_to_name.get(cat_id, str(cat_id))
        cls_id = class_name_to_id.get(name, cat_id_to_fallback.get(cat_id, 0))
        x, y, bw, bh = [float(v) for v in bbox]
        xc = min(max((x + bw / 2.0) / w_img, 0.0), 1.0)
        yc = min(max((y + bh / 2.0) / h_img, 0.0), 1.0)
        bw_n = min(max(bw / w_img, 1e-6), 1.0)
        bh_n = min(max(bh / h_img, 1e-6), 1.0)
        line = f"{cls_id} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}"

        file_name = str(meta.get("file_name", "")).strip()
        if not file_name:
            continue
        by_file.setdefault(file_name, []).append(line)
        by_tail.setdefault(_tail_key(file_name), []).append(line)

    out: Dict[str, List[str]] = {}
    out.update(by_file)
    for key, val in by_tail.items():
        out[f"TAIL::{key}"] = val
    return out


def _find_labels_for_image(
    src_root: Path,
    img_path: Path,
    coco_map: Dict[str, List[str]],
) -> Tuple[List[str], str, bool]:
    # 1) Standard YOLO layout: .../images/{split}/xxx.jpg -> .../labels/{split}/xxx.txt
    parts = [p.lower() for p in img_path.parts]
    if "images" in parts:
        idx = parts.index("images")
        if idx + 1 < len(parts):
            split_name = parts[idx + 1]
            if split_name in {"train", "val", "test"}:
                lbl_path = src_root / "labels" / split_name / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
                    invalid = any(not _line_is_valid(ln) for ln in lines)
                    return (lines if not invalid else [], "labels_dir", invalid)
                return ([], "labels_dir_missing", False)

    # 2) Flat train + COCO
    if img_path.parent.name.lower() == "train":
        direct = coco_map.get(img_path.name, [])
        if direct:
            return (direct, "coco_json", False)
        tail = _tail_key(img_path.name)
        by_tail = coco_map.get(f"TAIL::{tail}", [])
        if by_tail:
            return (by_tail, "coco_json_tail", False)

    # 3) Adjacent txt fallback
    adjacent = img_path.with_suffix(".txt")
    if adjacent.exists():
        lines = [ln.strip() for ln in adjacent.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        invalid = any(not _line_is_valid(ln) for ln in lines)
        return (lines if not invalid else [], "adjacent_txt", invalid)

    return ([], "no_label", False)


def _split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("sum(train_ratio, val_ratio, test_ratio) must be > 0")
    tr = train_ratio / total
    vr = val_ratio / total
    te = test_ratio / total

    n_train = int(n * tr)
    n_val = int(n * vr)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
    return n_train, n_val, n_test


def main() -> int:
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    if not src_root.exists():
        raise FileNotFoundError(f"src-root not found: {src_root}")

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    names = _load_names(Path(args.class_yaml))
    class_name_to_id = {v: k for k, v in names.items()}
    coco_map = _build_coco_label_maps(src_root, class_name_to_id)

    all_images = _iter_all_images(src_root)
    seen: set[str] = set()
    records: List[ImageRecord] = []
    invalid_label_images: List[str] = []
    source_counter: Dict[str, int] = {}

    for img in all_images:
        rid = str(img.resolve()).lower()
        if rid in seen:
            continue
        seen.add(rid)

        lines, source, invalid = _find_labels_for_image(src_root, img, coco_map)
        if invalid:
            invalid_label_images.append(str(img).replace("\\", "/"))
            if args.drop_invalid_label:
                continue
            lines = []
            source = f"{source}_invalid_dropped"
        records.append(ImageRecord(src_img=img, label_lines=lines, label_source=source))
        source_counter[source] = source_counter.get(source, 0) + 1

    rng = random.Random(args.seed)
    rng.shuffle(records)

    n_total = len(records)
    n_train, n_val, _n_test = _split_counts(n_total, args.train_ratio, args.val_ratio, args.test_ratio)
    train_rows = records[:n_train]
    val_rows = records[n_train:n_train + n_val]
    test_rows = records[n_train + n_val:]

    splits = [("train", train_rows), ("val", val_rows), ("test", test_rows)]
    mapping_rows: List[Dict[str, str]] = []
    split_stats: Dict[str, Dict[str, int]] = {}

    for split_name, rows in splits:
        img_dir = out_root / "images" / split_name
        lbl_dir = out_root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        labeled = 0
        unlabeled = 0
        boxes = 0
        for rec in rows:
            key = hashlib.md5(rec.src_img.as_posix().encode("utf-8")).hexdigest()[:8]
            stem = f"{rec.src_img.stem}__{key}"
            img_dst = img_dir / f"{stem}{rec.src_img.suffix.lower()}"
            lbl_dst = lbl_dir / f"{stem}.txt"

            _safe_link_or_copy(rec.src_img, img_dst, copy_images=args.copy_images)
            lbl_dst.write_text("\n".join(rec.label_lines), encoding="utf-8")

            is_labeled = len(rec.label_lines) > 0
            if is_labeled:
                labeled += 1
                boxes += len(rec.label_lines)
            else:
                unlabeled += 1
            mapping_rows.append(
                {
                    "split": split_name,
                    "src_image": str(rec.src_img).replace("\\", "/"),
                    "dst_image": str(img_dst).replace("\\", "/"),
                    "dst_label": str(lbl_dst).replace("\\", "/"),
                    "label_source": rec.label_source,
                    "is_labeled": "1" if is_labeled else "0",
                    "box_count": str(len(rec.label_lines)),
                }
            )
        split_stats[split_name] = {
            "images": len(rows),
            "labeled_images": labeled,
            "unlabeled_images": unlabeled,
            "boxes": boxes,
        }

    # If val split disabled, point val to test for training compatibility.
    val_ref = "images/val" if len(val_rows) > 0 else "images/test"
    ds_yaml = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": val_ref,
        "test": "images/test",
        "names": {k: v for k, v in sorted(names.items(), key=lambda kv: kv[0])},
    }
    (out_root / "dataset.yaml").write_text(
        yaml.safe_dump(ds_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    mapping_csv = out_root / "source_to_dataset_mapping.csv"
    with mapping_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "split",
            "src_image",
            "dst_image",
            "dst_label",
            "label_source",
            "is_labeled",
            "box_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in mapping_rows:
            writer.writerow(row)

    report = {
        "src_root": str(src_root).replace("\\", "/"),
        "out_root": str(out_root).replace("\\", "/"),
        "seed": args.seed,
        "ratios": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "totals": {
            "all_images_found": len(all_images),
            "unique_images_used": n_total,
            "invalid_label_images": len(invalid_label_images),
        },
        "splits": split_stats,
        "label_source_counts": source_counter,
        "invalid_label_examples": invalid_label_images[:50],
        "dataset_yaml": str((out_root / "dataset.yaml")).replace("\\", "/"),
        "mapping_csv": str(mapping_csv).replace("\\", "/"),
    }
    report_path = out_root / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

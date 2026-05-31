from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and rebuild a standardized YOLO dataset with 8:2 split."
    )
    parser.add_argument("--src-root", default="data/interim/frames")
    parser.add_argument("--class-yaml", default="data/processed/yolo_dataset/dataset.yaml")
    parser.add_argument("--out-root", default="data/processed/yolo_dataset_std_80_20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--include-coco-train",
        action="store_true",
        help="Also include src-root/train with _annotations.coco.json.",
    )
    return parser.parse_args()


def _load_names(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = raw.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _iter_images(root: Path) -> List[Path]:
    all_imgs: List[Path] = []
    for split in ("train", "val", "test"):
        split_dir = root / "images" / split
        if split_dir.exists():
            for p in split_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    all_imgs.append(p)
    return sorted(all_imgs)


def _iter_flat_train_images(root: Path) -> List[Path]:
    train_dir = root / "train"
    if not train_dir.exists():
        return []
    out: List[Path] = []
    for p in train_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def _label_for_image(src_root: Path, image_path: Path) -> Path:
    split_name = image_path.parent.name
    return src_root / "labels" / split_name / f"{image_path.stem}.txt"


def _load_coco_label_map(src_root: Path, class_name_to_id: Dict[str, int]) -> Dict[str, List[str]]:
    """Load COCO bbox annotations in src_root/train/_annotations.coco.json and convert to YOLO lines."""
    ann_path = src_root / "train" / "_annotations.coco.json"
    if not ann_path.exists():
        return {}

    raw = json.loads(ann_path.read_text(encoding="utf-8"))
    images = raw.get("images", [])
    anns = raw.get("annotations", [])
    cats = raw.get("categories", [])

    image_meta: Dict[int, Dict] = {int(i["id"]): i for i in images if "id" in i}
    cat_id_to_name: Dict[int, str] = {int(c["id"]): str(c.get("name", c["id"])) for c in cats}
    # Fallback to contiguous cat index when class name missing in target schema.
    cat_ids_sorted = sorted(cat_id_to_name.keys())
    cat_id_to_fallback = {cid: idx for idx, cid in enumerate(cat_ids_sorted)}

    by_file: Dict[str, List[str]] = {}
    by_tail: Dict[str, List[str]] = {}

    def _tail_key(name: str) -> str:
        s = name.lower()
        marker = "__f"
        idx = s.find(marker)
        return s[idx:] if idx >= 0 else s
    for a in anns:
        img_id = int(a.get("image_id", -1))
        cat_id = int(a.get("category_id", -1))
        bbox = a.get("bbox", None)  # COCO: x, y, w, h (pixel)
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
        xc = (x + bw / 2.0) / w_img
        yc = (y + bh / 2.0) / h_img
        bw_n = bw / w_img
        bh_n = bh / h_img
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        bw_n = min(max(bw_n, 1e-6), 1.0)
        bh_n = min(max(bh_n, 1e-6), 1.0)

        file_name = str(meta.get("file_name", ""))
        if not file_name:
            continue
        line = f"{cls_id} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}"
        by_file.setdefault(file_name, []).append(line)
        by_tail.setdefault(_tail_key(file_name), []).append(line)
    # merge maps in one dict-like namespace
    out: Dict[str, List[str]] = {}
    out.update(by_file)
    for k, v in by_tail.items():
        out[f"TAIL::{k}"] = v
    return out


def _line_is_valid(line: str) -> bool:
    t = line.strip().split()
    if len(t) < 5:
        return False
    try:
        [float(x) for x in t]
    except Exception:
        return False
    return True


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


def _count_boxes(lbl_dir: Path) -> Tuple[int, Counter]:
    boxes = 0
    cls_counter: Counter = Counter()
    if not lbl_dir.exists():
        return boxes, cls_counter
    for f in lbl_dir.glob("*.txt"):
        for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = ln.strip()
            if not s:
                continue
            t = s.split()
            if len(t) < 5:
                continue
            boxes += 1
            cls_counter[int(float(t[0]))] += 1
    return boxes, cls_counter


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
    images = _iter_images(src_root)
    flat_images = _iter_flat_train_images(src_root) if args.include_coco_train else []
    coco_map = _load_coco_label_map(src_root, class_name_to_id) if args.include_coco_train else {}

    missing_label: List[str] = []
    invalid_label_files: List[str] = []
    valid_pairs: List[Tuple[Path, Optional[Path], Optional[List[str]]]] = []

    for img in images:
        lbl = _label_for_image(src_root, img)
        if not lbl.exists():
            missing_label.append(str(img).replace("\\", "/"))
            continue
        lines = lbl.read_text(encoding="utf-8", errors="ignore").splitlines()
        bad = [ln for ln in lines if ln.strip() and not _line_is_valid(ln)]
        if bad:
            invalid_label_files.append(str(lbl).replace("\\", "/"))
            continue
        valid_pairs.append((img, lbl, None))

    # Optional flat train + COCO annotations
    for img in flat_images:
        tail = img.name.lower()
        idx = tail.find("__f")
        tail = tail[idx:] if idx >= 0 else tail
        lines = coco_map.get(img.name, []) or coco_map.get(f"TAIL::{tail}", [])
        if not lines:
            missing_label.append(str(img).replace("\\", "/"))
            continue
        bad = [ln for ln in lines if ln.strip() and not _line_is_valid(ln)]
        if bad:
            invalid_label_files.append(str(img).replace("\\", "/"))
            continue
        valid_pairs.append((img, None, lines))

    rng = random.Random(args.seed)
    rng.shuffle(valid_pairs)
    n = len(valid_pairs)
    n_train = int(n * 0.8)
    train_pairs = valid_pairs[:n_train]
    val_pairs = valid_pairs[n_train:]

    for split, pairs in (("train", train_pairs), ("val", val_pairs)):
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_src, lbl_src, coco_lines in pairs:
            rel_key = str(img_src.as_posix()).encode("utf-8")
            short = hashlib.md5(rel_key).hexdigest()[:8]
            out_stem = f"{img_src.stem}__{short}"
            img_dst = img_dir / f"{out_stem}{img_src.suffix.lower()}"
            lbl_dst = lbl_dir / f"{out_stem}.txt"
            _safe_link_or_copy(img_src, img_dst, copy_images=args.copy_images)
            if lbl_src is not None:
                shutil.copy2(lbl_src, lbl_dst)
            else:
                lbl_dst.write_text("\n".join(coco_lines or []), encoding="utf-8")

    ds_yaml = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {k: v for k, v in sorted(names.items(), key=lambda kv: kv[0])},
    }
    (out_root / "dataset.yaml").write_text(
        yaml.safe_dump(ds_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    train_boxes, train_cls = _count_boxes(out_root / "labels" / "train")
    val_boxes, val_cls = _count_boxes(out_root / "labels" / "val")

    report = {
        "src_root": str(src_root).replace("\\", "/"),
        "out_root": str(out_root).replace("\\", "/"),
        "audit": {
            "total_images_found": len(images),
            "flat_train_images_found": len(flat_images),
            "valid_image_label_pairs": len(valid_pairs),
            "missing_label_images": len(missing_label),
            "invalid_label_files": len(invalid_label_files),
        },
        "split": {
            "ratio": "8:2",
            "train_images": len(train_pairs),
            "val_images": len(val_pairs),
            "test_images": 0,
            "train_boxes": train_boxes,
            "val_boxes": val_boxes,
            "test_boxes": 0,
        },
        "class_distribution": {
            "train": {names.get(k, str(k)): v for k, v in sorted(train_cls.items())},
            "val": {names.get(k, str(k)): v for k, v in sorted(val_cls.items())},
        },
        "examples": {
            "missing_label_images": missing_label[:20],
            "invalid_label_files": invalid_label_files[:20],
        },
    }
    report_path = out_root / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

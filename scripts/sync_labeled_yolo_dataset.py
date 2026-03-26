from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync labeled YOLO dataset (images/labels train/val) into processed training directory."
    )
    parser.add_argument("--src-root", default="data/interim/labeling/frames")
    parser.add_argument("--out-root", default="data/processed/yolo_dataset")
    parser.add_argument(
        "--coco-json",
        default="data/interim/labeling/frames/train/_annotations.coco.json",
        help="Used to infer class names for dataset.yaml",
    )
    parser.add_argument("--clean", action="store_true", help="Clean output root before sync")
    return parser.parse_args()


def _copy_tree(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in src.glob("*"):
        if p.is_file():
            shutil.copy2(p, dst / p.name)
            copied += 1
    return copied


def _read_classes(coco_json: Path) -> Dict[int, str]:
    if not coco_json.exists():
        return {}
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    cats = data.get("categories", [])
    names = {}
    for c in cats:
        try:
            cid = int(c["id"])
            cname = str(c["name"])
            names[cid] = cname
        except Exception:
            continue
    return names


def _safe_names(names: Dict[int, str]) -> Dict[int, str]:
    if not names:
        return {0: "object"}
    # Keep all classes to remain index-consistent with YOLO label txt.
    return names


def main() -> int:
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)

    src_img_train = src_root / "images" / "train"
    src_img_val = src_root / "images" / "val"
    src_lbl_train = src_root / "labels" / "train"
    src_lbl_val = src_root / "labels" / "val"

    if not src_img_train.exists() or not src_lbl_train.exists():
        raise FileNotFoundError(
            f"missing train images/labels under {src_root}. Expected {src_img_train} and {src_lbl_train}"
        )

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    out_img_train = out_root / "images" / "train"
    out_img_val = out_root / "images" / "val"
    out_lbl_train = out_root / "labels" / "train"
    out_lbl_val = out_root / "labels" / "val"

    n_img_train = _copy_tree(src_img_train, out_img_train)
    n_lbl_train = _copy_tree(src_lbl_train, out_lbl_train)
    n_img_val = _copy_tree(src_img_val, out_img_val) if src_img_val.exists() else 0
    n_lbl_val = _copy_tree(src_lbl_val, out_lbl_val) if src_lbl_val.exists() else 0

    class_map = _safe_names(_read_classes(Path(args.coco_json)))
    class_map = {int(k): v for k, v in sorted(class_map.items(), key=lambda x: x[0])}

    dataset_yaml = out_root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_root.resolve()).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val" if n_img_val > 0 else "images/train",
                "names": class_map,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "src_root": str(src_root).replace("\\", "/"),
        "out_root": str(out_root).replace("\\", "/"),
        "images_train": n_img_train,
        "labels_train": n_lbl_train,
        "images_val": n_img_val,
        "labels_val": n_lbl_val,
        "class_count": len(class_map),
        "classes": class_map,
        "dataset_yaml": str(dataset_yaml).replace("\\", "/"),
    }
    (out_root / "sync_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

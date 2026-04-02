from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build focused pose dataset by oversampling target classes.")
    parser.add_argument("--src-dataset-yaml", default="data/processed/yolo_pose_dataset_std_80_20/dataset.yaml")
    parser.add_argument("--out-root", default="data/processed/yolo_pose_dataset_focus_v1")
    parser.add_argument(
        "--focus-class-multiplier",
        nargs="*",
        default=["lab_coat:2", "gloved_hand:3", "spatula:8"],
        help='Format: class_name:extra_repeats. Example: "lab_coat:2 gloved_hand:3".',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def _parse_multipliers(values: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in values:
        if ":" not in item:
            continue
        name, n = item.split(":", 1)
        name = name.strip()
        try:
            rep = int(n)
        except ValueError:
            continue
        if name and rep > 0:
            out[name] = rep
    return out


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _label_classes(label_path: Path) -> Set[int]:
    cls_ids: Set[int] = set()
    if not label_path.exists():
        return cls_ids
    for line in label_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        toks = s.split()
        if not toks:
            continue
        try:
            cid = int(float(toks[0]))
        except ValueError:
            continue
        cls_ids.add(cid)
    return cls_ids


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    src_yaml = Path(args.src_dataset_yaml)
    if not src_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {src_yaml}")
    cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))

    src_root = Path(cfg["path"])
    train_rel = Path(str(cfg.get("train", "images/train")).replace("\\", "/"))
    val_rel = Path(str(cfg.get("val", "images/val")).replace("\\", "/"))
    train_images = train_rel if train_rel.is_absolute() else (src_root / train_rel)
    val_images = val_rel if val_rel.is_absolute() else (src_root / val_rel)
    train_split = train_rel.parts[-1] if train_rel.parts else "train"
    val_split = val_rel.parts[-1] if val_rel.parts else "val"
    train_labels = src_root / "labels" / train_split
    val_labels = src_root / "labels" / val_split

    out_root = Path(args.out_root)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    names_raw = cfg.get("names", {})
    names: Dict[int, str] = {int(k): str(v) for k, v in names_raw.items()}
    name_to_id: Dict[str, int] = {v: k for k, v in names.items()}

    requested = _parse_multipliers(args.focus_class_multiplier)
    focus_class_to_rep: Dict[int, int] = {}
    for cname, rep in requested.items():
        if cname in name_to_id:
            focus_class_to_rep[name_to_id[cname]] = rep

    train_imgs = sorted([p for p in train_images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    base_copied = 0
    repeated_copied = 0
    per_class_repeat_hits: Dict[str, int] = {names[cid]: 0 for cid in focus_class_to_rep.keys()}

    for img_path in train_imgs:
        label_path = train_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        _safe_copy(img_path, out_root / "images" / "train" / img_path.name)
        _safe_copy(label_path, out_root / "labels" / "train" / label_path.name)
        base_copied += 1

        cls_ids = _label_classes(label_path)
        extra_rep = 0
        for cid in cls_ids:
            extra_rep = max(extra_rep, focus_class_to_rep.get(cid, 0))
        if extra_rep <= 0:
            continue

        for cid in cls_ids:
            if cid in focus_class_to_rep:
                per_class_repeat_hits[names[cid]] += 1

        for rep_id in range(1, extra_rep + 1):
            suffix = f"__focusrep{rep_id}"
            new_img_name = f"{img_path.stem}{suffix}{img_path.suffix}"
            new_lbl_name = f"{label_path.stem}{suffix}{label_path.suffix}"
            _safe_copy(img_path, out_root / "images" / "train" / new_img_name)
            _safe_copy(label_path, out_root / "labels" / "train" / new_lbl_name)
            repeated_copied += 1

    # Val set keeps original distribution for fair evaluation.
    val_copied = 0
    val_imgs = sorted([p for p in val_images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for img_path in val_imgs:
        label_path = val_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        _safe_copy(img_path, out_root / "images" / "val" / img_path.name)
        _safe_copy(label_path, out_root / "labels" / "val" / label_path.name)
        val_copied += 1

    out_cfg = dict(cfg)
    out_cfg["path"] = str(out_root.resolve()).replace("\\", "/")
    out_cfg["train"] = "images/train"
    out_cfg["val"] = "images/val"
    if "test" in out_cfg:
        out_cfg["test"] = "images/val"
    (out_root / "dataset.yaml").write_text(yaml.safe_dump(out_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    report = {
        "src_dataset_yaml": str(src_yaml),
        "out_dataset_yaml": str((out_root / "dataset.yaml").resolve()),
        "focus_class_multiplier": requested,
        "resolved_focus_class_multiplier": {names[k]: v for k, v in focus_class_to_rep.items()},
        "train_base_copied": base_copied,
        "train_repeated_copied": repeated_copied,
        "train_total_after_focus": base_copied + repeated_copied,
        "val_copied": val_copied,
        "per_class_repeat_hits": per_class_repeat_hits,
    }
    (out_root / "focus_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dataset yaml: {out_root / 'dataset.yaml'}")
    print(f"train base: {base_copied}")
    print(f"train repeated: {repeated_copied}")
    print(f"train total: {base_copied + repeated_copied}")
    print(f"val total: {val_copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

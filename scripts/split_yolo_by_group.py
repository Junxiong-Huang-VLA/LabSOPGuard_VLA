from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_dataset_yaml(src_root: Path) -> Dict[str, object]:
    path = src_root / "dataset.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_dataset_yaml(out_root: Path, source_payload: Dict[str, object]) -> None:
    names = source_payload.get("names", {})
    nc = source_payload.get("nc", len(names) if isinstance(names, (list, dict)) else 0)
    payload = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": nc,
        "names": names,
    }
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    except Exception:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    (out_root / "dataset.yaml").write_text(text, encoding="utf-8")


def _load_metadata(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidates = [
                row.get("image_path"),
                row.get("frame_path"),
                row.get("path"),
                row.get("filename"),
                row.get("file_name"),
            ]
            for value in candidates:
                if not value:
                    continue
                normalized = Path(value).as_posix()
                rows[normalized] = row
                rows[Path(value).name] = row
    return rows


def _discover_images(src_root: Path) -> List[Path]:
    image_root = src_root / "images"
    if not image_root.exists():
        raise FileNotFoundError(f"Missing YOLO images directory: {image_root}")
    return sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _relative_image_key(src_root: Path, image_path: Path) -> str:
    try:
        return image_path.relative_to(src_root).as_posix()
    except ValueError:
        return image_path.as_posix()


def _label_path_for_image(src_root: Path, image_path: Path) -> Path:
    rel = image_path.relative_to(src_root / "images")
    direct = src_root / "labels" / rel.with_suffix(".txt")
    if direct.exists():
        return direct
    parts = rel.parts
    if parts and parts[0] in {"train", "val", "test", "all"}:
        compact = src_root / "labels" / Path(*parts[1:]).with_suffix(".txt")
        if compact.exists():
            return compact
    return direct


def _safe_group_text(value: str) -> str:
    value = value.strip().replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9_.:/-]+", "_", value)
    return value or "unknown"


def _infer_group_from_path(src_root: Path, image_path: Path, group_regex: Optional[re.Pattern[str]]) -> str:
    rel = _relative_image_key(src_root, image_path)
    text = rel.lower()
    if group_regex:
        match = group_regex.search(rel)
        if match:
            if match.groupdict():
                return "::".join(f"{key}={_safe_group_text(value)}" for key, value in match.groupdict().items() if value)
            return "::".join(_safe_group_text(item) for item in match.groups() if item)
    camera_match = re.search(r"(camera[_-]?\d+|cam[_-]?\d+|usb[_-]?\d+|top|bottom|front|side|left|right)", text)
    camera_id = camera_match.group(1) if camera_match else ""
    rel_parts = Path(rel).parts
    content_parts = [part for part in rel_parts if part not in {"images", "train", "val", "test", "all"}]
    parent_key = "/".join(content_parts[:-1]) if len(content_parts) > 1 else image_path.stem
    stem_key = re.sub(r"([_-]?(frame|img|image)[_-]?\d+)$", "", image_path.stem, flags=re.IGNORECASE)
    video_key = parent_key if parent_key and parent_key != "." else stem_key
    pieces = [piece for piece in [camera_id, video_key] if piece]
    return "::".join(_safe_group_text(piece) for piece in pieces) or _safe_group_text(image_path.parent.name)


def _lookup_metadata_row(metadata: Dict[str, Dict[str, str]], src_root: Path, image_path: Path) -> Optional[Dict[str, str]]:
    keys = [
        _relative_image_key(src_root, image_path),
        image_path.relative_to(src_root / "images").as_posix(),
        image_path.name,
    ]
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _group_for_image(
    src_root: Path,
    image_path: Path,
    metadata: Dict[str, Dict[str, str]],
    group_columns: List[str],
    group_regex: Optional[re.Pattern[str]],
) -> str:
    row = _lookup_metadata_row(metadata, src_root, image_path)
    if row:
        pairs = []
        for column in group_columns:
            value = row.get(column, "").strip()
            if value:
                pairs.append((column, value))
        if pairs:
            return "::".join(f"{column}={_safe_group_text(value)}" for column, value in pairs)
    return _infer_group_from_path(src_root, image_path, group_regex)


def _assign_groups(groups: Dict[str, List[Path]], *, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    total = sum(len(paths) for _, paths in items)
    targets = {"test": total * test_ratio, "val": total * val_ratio}
    counts = {"train": 0, "val": 0, "test": 0}
    assignment: Dict[str, str] = {}
    for group, paths in sorted(items, key=lambda item: len(item[1]), reverse=True):
        deficits = {split: targets[split] - counts[split] for split in ("test", "val")}
        split = max(deficits, key=lambda key: deficits[key])
        if deficits[split] <= 0:
            split = "train"
        assignment[group] = split
        counts[split] += len(paths)
    return assignment


def _link_or_copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
        return
    if mode == "symlink":
        os.symlink(source.resolve(), target)
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_split(
    src_root: Path,
    out_root: Path,
    groups: Dict[str, List[Path]],
    assignment: Dict[str, str],
    mode: str,
) -> Dict[str, object]:
    summary: Dict[str, object] = {"splits": {}, "groups": {}}
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        summary["splits"][split] = {"images": 0, "groups": 0, "missing_labels": 0}
    for group, images in groups.items():
        split = assignment[group]
        summary["splits"][split]["groups"] += 1
        summary["groups"][group] = {"split": split, "images": len(images)}
        for image_path in images:
            rel = image_path.relative_to(src_root / "images")
            if rel.parts and rel.parts[0] in {"train", "val", "test", "all"}:
                rel = Path(*rel.parts[1:])
            image_target = out_root / "images" / split / rel
            label_source = _label_path_for_image(src_root, image_path)
            label_target = (out_root / "labels" / split / rel).with_suffix(".txt")
            _link_or_copy(image_path, image_target, mode)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            if label_source.exists():
                _link_or_copy(label_source, label_target, mode)
            else:
                label_target.write_text("", encoding="utf-8")
                summary["splits"][split]["missing_labels"] += 1
            summary["splits"][split]["images"] += 1
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a YOLO dataset by whole camera/experiment/video groups.")
    parser.add_argument("--src-root", required=True, type=Path, help="Source YOLO dataset root containing images/ and labels/.")
    parser.add_argument("--out-root", required=True, type=Path, help="Output YOLO dataset root to create.")
    parser.add_argument("--metadata-csv", type=Path, default=None, help="Optional CSV with image_path/frame_path and group columns.")
    parser.add_argument("--group-by", default="camera_id,experiment_id,video_id", help="Comma-separated metadata columns used as group key.")
    parser.add_argument("--group-regex", default=None, help="Optional regex applied to image relative path; captures form the group key.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_root = args.src_root.resolve()
    out_root = args.out_root.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Output directory exists and is not empty: {out_root}")
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Require val_ratio >= 0, test_ratio >= 0, and val_ratio + test_ratio < 1.")

    metadata = _load_metadata(args.metadata_csv)
    group_columns = [item.strip() for item in args.group_by.split(",") if item.strip()]
    group_regex = re.compile(args.group_regex) if args.group_regex else None
    images = _discover_images(src_root)
    groups: Dict[str, List[Path]] = defaultdict(list)
    for image_path in images:
        group = _group_for_image(src_root, image_path, metadata, group_columns, group_regex)
        groups[group].append(image_path)
    assignment = _assign_groups(groups, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    source_payload = _read_dataset_yaml(src_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = _copy_split(src_root, out_root, groups, assignment, args.mode)
    summary["source_root"] = str(src_root)
    summary["output_root"] = str(out_root)
    summary["group_count"] = len(groups)
    summary["image_count"] = len(images)
    summary["group_by"] = group_columns
    _write_dataset_yaml(out_root, source_payload)
    (out_root / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

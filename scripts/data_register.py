from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SUPPORTED_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".npy",
    ".npz",
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
}


def build_maps(
    files: Iterable[Path],
    rgb_suffix: str,
    depth_suffix: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    rgb_pat = re.compile(rf"(.+?){re.escape(rgb_suffix)}$", re.IGNORECASE)
    depth_pat = re.compile(rf"(.+?){re.escape(depth_suffix)}$", re.IGNORECASE)
    rgb_map: Dict[str, str] = {}
    depth_map: Dict[str, str] = {}

    for path in files:
        stem = path.stem
        rgb_match = rgb_pat.match(stem)
        depth_match = depth_pat.match(stem)
        if rgb_match:
            rgb_map[rgb_match.group(1)] = str(path)
        if depth_match:
            depth_map[depth_match.group(1)] = str(path)

    return rgb_map, depth_map


def write_manifest(
    manifest_path: Path,
    rows: List[dict],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_id",
                "rgb_path",
                "depth_path",
                "has_rgb",
                "has_depth",
                "pair_status",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["sample_id"],
                    row["rgb_path"],
                    row["depth_path"],
                    row["has_rgb"],
                    row["has_depth"],
                    row["pair_status"],
                ]
            )


def build_rows(rgb_map: Dict[str, str], depth_map: Dict[str, str]) -> List[dict]:
    keys = sorted(set(rgb_map) | set(depth_map))
    rows: List[dict] = []
    for key in keys:
        rgb = rgb_map.get(key, "")
        depth = depth_map.get(key, "")
        has_rgb = int(bool(rgb))
        has_depth = int(bool(depth))
        status = "paired" if has_rgb and has_depth else "missing_pair"
        rows.append(
            {
                "sample_id": key,
                "rgb_path": rgb,
                "depth_path": depth,
                "has_rgb": has_rgb,
                "has_depth": has_depth,
                "pair_status": status,
            }
        )
    return rows


def update_dataset_yaml(
    dataset_yaml: Path,
    dataset_name: str,
    raw_root: str,
    manifest_csv: str,
    rgb_suffix: str,
    depth_suffix: str,
) -> None:
    content = (
        "dataset:\n"
        f"  name: {dataset_name}\n"
        f"  raw_root: {raw_root}\n"
        f"  manifest_csv: {manifest_csv}\n"
        "  modalities: [rgb, depth]\n"
        "  pairing:\n"
        f"    rgb_suffix: \"{rgb_suffix}\"\n"
        f"    depth_suffix: \"{depth_suffix}\"\n"
    )
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register RGB/Depth dataset and generate pair manifests."
    )
    parser.add_argument("--dataset-root", required=True, help="Raw dataset directory.")
    parser.add_argument(
        "--manifest-csv",
        default="data/interim/dataset_manifest.csv",
        help="All-sample manifest output path.",
    )
    parser.add_argument(
        "--paired-manifest-csv",
        default="data/interim/dataset_manifest_paired.csv",
        help="Paired-only manifest output path.",
    )
    parser.add_argument("--rgb-suffix", default="_rgb", help="RGB file stem suffix.")
    parser.add_argument("--depth-suffix", default="_depth", help="Depth file stem suffix.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan dataset root recursively.",
    )
    parser.add_argument(
        "--dataset-yaml",
        default="configs/data/dataset.yaml",
        help="Dataset config path for auto update.",
    )
    parser.add_argument(
        "--dataset-name",
        default="discription_pdf",
        help="Dataset name to write into dataset config.",
    )
    parser.add_argument(
        "--raw-root-for-config",
        default="data/raw/external/discription_pdf",
        help="raw_root value to write into dataset config.",
    )
    parser.add_argument(
        "--skip-config-update",
        action="store_true",
        help="Do not update configs/data/dataset.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if args.recursive:
        all_files = [p for p in dataset_root.rglob("*") if p.is_file()]
    else:
        all_files = [p for p in dataset_root.iterdir() if p.is_file()]
    files = [p for p in all_files if p.suffix.lower() in SUPPORTED_EXTS]

    rgb_map, depth_map = build_maps(files, args.rgb_suffix, args.depth_suffix)
    rows = build_rows(rgb_map, depth_map)
    paired_rows = [row for row in rows if row["pair_status"] == "paired"]

    manifest_csv = Path(args.manifest_csv)
    paired_manifest_csv = Path(args.paired_manifest_csv)
    write_manifest(manifest_csv, rows)
    write_manifest(paired_manifest_csv, paired_rows)

    if not args.skip_config_update:
        update_dataset_yaml(
            dataset_yaml=Path(args.dataset_yaml),
            dataset_name=args.dataset_name,
            raw_root=args.raw_root_for_config,
            manifest_csv=str(paired_manifest_csv).replace("\\", "/"),
            rgb_suffix=args.rgb_suffix,
            depth_suffix=args.depth_suffix,
        )

    print(f"dataset_root: {dataset_root}")
    print(f"manifest: {manifest_csv} (rows={len(rows)})")
    print(f"paired_manifest: {paired_manifest_csv} (rows={len(paired_rows)})")
    print(f"missing_pair: {len(rows) - len(paired_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

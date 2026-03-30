from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap pseudo labels into labeling_manifest.csv using YOLO.")
    parser.add_argument("--manifest-csv", default="data/interim/labeling/labeling_manifest.csv")
    parser.add_argument("--runtime-config", default="configs/model/detection_runtime.yaml")
    parser.add_argument(
        "--class-schema",
        default=None,
        help="Optional class schema yaml. If provided, labels not in schema classes will be dropped.",
    )
    parser.add_argument("--out-csv", default="data/interim/labeling/labeling_manifest_pseudo.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--device",
        default=None,
        help="Override runtime device, e.g. cpu/cuda:0. Defaults to runtime-config value.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    parser.add_argument(
        "--empty-status",
        default="pending",
        help="annotation_status for rows with zero pseudo boxes after filtering.",
    )
    return parser.parse_args()


def _load_runtime(path: Path) -> Dict:
    if not path.exists():
        return {"model": "yolov8n.pt", "device": "cuda:0", "class_registry": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _build_alias_map(class_registry: Dict[str, List[str]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for canonical, aliases in class_registry.items():
        m[canonical.lower()] = canonical
        for a in aliases:
            m[str(a).lower()] = canonical
    return m


def _load_schema_classes(path: Path | None) -> set[str] | None:
    if path is None or not path.exists():
        return None
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    classes = cfg.get("classes", [])
    names = {str(c.get("name", "")).strip() for c in classes if str(c.get("name", "")).strip()}
    return names or None


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _safe_read_image(path: Path):
    if os.name == "nt":
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size > 0:
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        except Exception:
            pass
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest_csv)
    runtime_cfg = _load_runtime(Path(args.runtime_config))
    model_name = str(runtime_cfg.get("model", "yolov8n.pt"))
    device = str(args.device or runtime_cfg.get("device", "cuda:0"))
    alias_map = _build_alias_map(runtime_cfg.get("class_registry", {}))
    schema_classes = _load_schema_classes(Path(args.class_schema)) if args.class_schema else None

    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("ultralytics is required") from exc

    model = YOLO(model_name)
    rows = _read_rows(manifest)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    out_rows: List[Dict[str, str]] = []
    dropped_unknown = 0
    pseudo_rows = 0
    empty_rows = 0
    for row in rows:
        frame_path = Path(row.get("frame_path", ""))
        if not frame_path.is_absolute():
            frame_path = Path.cwd() / frame_path
        if not frame_path.exists():
            out_rows.append(row)
            continue

        img = _safe_read_image(frame_path)
        if img is None:
            out_rows.append(row)
            continue

        preds = model.predict(source=img, conf=args.conf, verbose=False, device=device)
        anns = []
        for p in preds:
            names = p.names
            for b in p.boxes:
                cls_id = int(b.cls.item())
                raw = str(names.get(cls_id, cls_id))
                canonical = alias_map.get(raw.lower(), raw)
                if schema_classes is not None and canonical not in schema_classes:
                    dropped_unknown += 1
                    continue
                bbox = [float(v) for v in b.xyxy[0].tolist()]
                anns.append(
                    {
                        "class_name": canonical,
                        "bbox": [round(v, 2) for v in bbox],
                        "score": round(float(b.conf.item()), 4),
                        "source": "pseudo_yolo",
                    }
                )

        new_row = dict(row)
        new_row["labels_json"] = json.dumps(anns, ensure_ascii=False)
        if anns:
            new_row["annotation_status"] = "pseudo"
            pseudo_rows += 1
        else:
            new_row["annotation_status"] = args.empty_status
            empty_rows += 1
        out_rows.append(new_row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fields = list(out_rows[0].keys()) if out_rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"pseudo manifest: {out_csv}")
    print(f"rows processed: {len(out_rows)}")
    print(f"pseudo rows: {pseudo_rows}")
    print(f"empty rows: {empty_rows}")
    print(f"dropped unknown labels: {dropped_unknown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

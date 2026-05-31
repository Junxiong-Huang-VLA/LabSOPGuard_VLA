from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

from .capability_gap_report import CAPABILITY_REQUIREMENTS
from .schemas import SessionManifest, read_jsonl, write_jsonl
from .time_alignment import global_time_to_local_sec, parse_time
from .yolo_detector import canonical_yolo_label


DEFAULT_RETRAIN_CLASSES = [
    "balance",
    "beaker",
    "gloved_hand",
    "lab_coat",
    "paper",
    "reagent_bottle",
    "sample_bottle",
    "sample_bottle_blue",
    "spatula",
    "tube",
    "tube_cap",
    "pipette_tip",
    "pipette",
    "container",
    "ppe_storage",
]

_CANDIDATE_SOURCE_PATHS = (
    Path("metadata") / "micro_segments.jsonl",
    Path("metadata") / "key_action_segments.jsonl",
    Path("metadata") / "vector_metadata.jsonl",
)

_MISSING_CLASS_HINTS: dict[str, list[str]] = {
    "liquid": ["liquid", "fluid", "solution", "pour", "transfer", "pipette", "beaker", "tube", "fill", "level"],
    "control": ["button", "knob", "dial", "display", "readout", "panel", "balance", "press", "adjust", "set", "record"],
    "container_state": ["open", "close", "closed", "cap", "lid", "cover", "container", "bottle", "tube", "seal"],
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _frame_times(start_sec: float, end_sec: float, samples_per_segment: int) -> list[float]:
    count = max(1, int(samples_per_segment))
    if count == 1 or end_sec <= start_sec:
        return [(start_sec + end_sec) / 2.0]
    return [start_sec + (end_sec - start_sec) * index / (count - 1) for index in range(count)]


def _source_for_view(manifest: SessionManifest, view: str):
    return manifest.videos.get(view)


def _extract_frame(video_path: Path, local_sec: float, output_path: Path) -> bool:
    try:
        import cv2
    except Exception:
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(local_sec)) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(output_path), frame))
    finally:
        cap.release()


def _bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    if width <= 0 or height <= 0 or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    return ((x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height, bw / width, bh / height)


def _nearest_row(rows: list[dict[str, Any]], view: str, session_sec: float, max_delta_sec: float) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if str(row.get("source_view") or row.get("view") or "") == view
        and abs(float(row.get("alignment_time_sec", row.get("local_time_sec", 0.0)) or 0.0) - session_sec) <= max_delta_sec
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(float(row.get("alignment_time_sec", row.get("local_time_sec", 0.0)) or 0.0) - session_sec))


def _prelabel_lines(row: dict[str, Any] | None, class_to_id: dict[str, int]) -> list[str]:
    if row is None:
        return []
    width = int(row.get("frame_width") or 0)
    height = int(row.get("frame_height") or 0)
    lines: list[str] = []
    for detection in row.get("detections") or []:
        if not isinstance(detection, dict):
            continue
        label = canonical_yolo_label(detection.get("label") or detection.get("raw_label"))
        if label not in class_to_id:
            continue
        yolo_box = _bbox_to_yolo(list(detection.get("bbox") or []), width, height)
        if yolo_box is None:
            continue
        lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(class_to_id[label], *yolo_box))
    return lines


def export_yolo_relabel_pack(
    session_dir: str | Path,
    *,
    ground_truth_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    yolo_rows_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    capability_gap_path: str | Path | None = None,
    missing_classes: list[str] | None = None,
    candidate_source_path: str | Path | None = None,
    views: list[str] | None = None,
    samples_per_segment: int = 3,
    nearest_yolo_delta_sec: float = 0.35,
    max_candidates_per_class: int = 20,
) -> dict[str, Any]:
    session = Path(session_dir)
    manifest_source = Path(manifest_path) if manifest_path else session / "manifest.json"
    manifest = _load_manifest_optional(manifest_source)
    source_rows, source_rows_path = _load_candidate_source_rows(session, ground_truth_path, candidate_source_path)
    yolo_rows_source = Path(yolo_rows_path) if yolo_rows_path else session / "cv_outputs" / "yolo_micro_frame_rows.jsonl"
    yolo_rows = read_jsonl(yolo_rows_source) if yolo_rows_source.exists() else []
    target = Path(output_dir) if output_dir else session / "annotation" / "yolo_relabel_pack"
    selected_views = views or ["first_person", "third_person"]
    target_classes = _target_missing_classes(capability_gap_path, missing_classes)
    classes = _merge_classes(DEFAULT_RETRAIN_CLASSES, target_classes)
    class_to_id = {label: index for index, label in enumerate(classes)}
    candidate_seeds = _missing_class_candidate_seeds(
        source_rows,
        target_classes,
        max_candidates_per_class=max_candidates_per_class,
    )
    seeds_by_row_key: dict[str, list[dict[str, Any]]] = {}
    for seed in candidate_seeds:
        seeds_by_row_key.setdefault(str(seed["source_row_key"]), []).append(seed)

    review_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    image_count = 0
    prelabel_count = 0
    empty_template_count = 0
    failed_extracts: list[dict[str, Any]] = []
    for gt_index, gt in enumerate(source_rows, start=1):
        source_row_key = _source_row_key(gt, gt_index)
        start_sec = float(gt.get("start_sec", 0.0) or 0.0)
        end_sec = float(gt.get("end_sec", start_sec) or start_sec)
        primary = canonical_yolo_label(gt.get("primary_object"))
        for sample_index, session_sec in enumerate(_frame_times(start_sec, end_sec, samples_per_segment), start=1):
            for view in selected_views:
                source = _source_for_view(manifest, view) if manifest is not None else None
                local_sec = float(session_sec)
                video_path: Path | None = None
                if manifest is not None:
                    if source is None:
                        continue
                    global_time = parse_time(manifest.session_start_time) + timedelta(seconds=session_sec)
                    video_path = Path(source.path)
                    local_sec = global_time_to_local_sec(source, global_time)
                stem = f"{gt.get('micro_segment_id', f'gt_{gt_index:03d}')}_{view}_{sample_index:02d}"
                image_rel = Path("images") / "train" / f"{stem}.jpg"
                label_rel = Path("labels") / "train" / f"{stem}.txt"
                image_path = target / image_rel
                label_path = target / label_rel
                extracted = _extract_frame(video_path, local_sec, image_path) if video_path is not None else False
                if not extracted:
                    failed_extracts.append({"gt_id": gt.get("micro_segment_id"), "view": view, "local_sec": local_sec, "video_path": str(video_path) if video_path else None})
                else:
                    nearest = _nearest_row(yolo_rows, view, session_sec, nearest_yolo_delta_sec)
                    prelabels = _prelabel_lines(nearest, class_to_id)
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text("\n".join(prelabels) + ("\n" if prelabels else ""), encoding="utf-8")
                    image_count += 1
                    prelabel_count += len(prelabels)
                    review_rows.append(
                        {
                            "image": str(image_rel).replace("\\", "/"),
                            "label": str(label_rel).replace("\\", "/"),
                            "view": view,
                            "gt_micro_segment_id": gt.get("micro_segment_id"),
                            "source_prediction_id": gt.get("source_prediction_id"),
                            "session_sec": round(float(session_sec), 6),
                            "local_sec": round(float(local_sec), 6),
                            "expected_primary_object": primary,
                            "expected_interaction_type": gt.get("interaction_type"),
                            "expected_action_type": gt.get("action_type"),
                            "prelabel_count": len(prelabels),
                            "nearest_yolo_row_time_sec": nearest.get("alignment_time_sec") if nearest else None,
                            "requires_human_bbox_review": True,
                        }
                    )
                for seed in seeds_by_row_key.get(source_row_key, []):
                    template_rel = Path("empty_annotation_templates") / str(seed["target_class"]) / f"{stem}.txt"
                    template_path = target / template_rel
                    template_path.parent.mkdir(parents=True, exist_ok=True)
                    template_path.write_text("", encoding="utf-8")
                    empty_template_count += 1
                    candidate_rows.append(
                        {
                            "candidate_id": f"{stem}:{seed['target_class']}",
                            "target_class": seed["target_class"],
                            "target_class_id": class_to_id.get(str(seed["target_class"])),
                            "candidate_status": "candidate_unreviewed",
                            "auto_confirmed": False,
                            "can_upgrade_confirmed": False,
                            "requires_human_bbox_review": True,
                            "image": str(image_rel).replace("\\", "/"),
                            "image_extracted": bool(extracted),
                            "label": str(label_rel).replace("\\", "/"),
                            "empty_template": str(template_rel).replace("\\", "/"),
                            "view": view,
                            "source_row_key": source_row_key,
                            "source_row_id": seed.get("source_row_id"),
                            "source_prediction_id": gt.get("source_prediction_id"),
                            "session_sec": round(float(session_sec), 6),
                            "local_sec": round(float(local_sec), 6),
                            "candidate_score": seed.get("candidate_score"),
                            "candidate_reason": seed.get("candidate_reason"),
                            "expected_primary_object": primary,
                            "expected_interaction_type": gt.get("interaction_type"),
                            "expected_action_type": gt.get("action_type"),
                        }
                    )

    write_jsonl(target / "review_manifest.jsonl", review_rows)
    write_jsonl(target / "candidate_manifest.jsonl", candidate_rows)
    (target / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    dataset_yaml = {
        "path": str(target.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/train",
        "names": {index: label for index, label in enumerate(classes)},
    }
    _write_json(target / "dataset.yaml.json", dataset_yaml)
    (target / "dataset.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/train\nnames:\n".format(str(target.resolve()).replace("\\", "/"))
        + "".join(f"  {index}: {label}\n" for index, label in enumerate(classes)),
        encoding="utf-8",
    )
    notes = [
        "# YOLO Relabel Pack",
        "",
        "This pack is generated from full-window micro GT time ranges.",
        "The `.txt` labels are prelabels from the currently configured YOLO rows and must be reviewed before training.",
        "Do not treat this as final GT until bbox labels are corrected.",
        "Missing-class candidates are a worklist only and do not automatically upgrade any event, micro-segment, or process step to confirmed.",
        "",
        "Target missing classes:",
        "",
        *[f"- {class_name}" for class_name in target_classes],
        "",
        "Generated review files:",
        "",
        "- `review_manifest.jsonl`: extracted image/prelabel rows for bbox review.",
        "- `candidate_manifest.jsonl`: missing-class candidate rows with candidate_unreviewed status.",
        "- `empty_annotation_templates/`: empty per-class YOLO label templates for manual boxes.",
        "",
        "Suggested train command after bbox review:",
        "",
        "```powershell",
        "python LabSOPGuard/scripts/train_yolo_lab.py --dataset-yaml \"{}\" --model \"{}\" --epochs 50 --imgsz 960 --batch 8 --device cpu --project LabSOPGuard/outputs/training --name yolo26s_lab_full_window_relabel".format(
            str((target / "dataset.yaml").resolve()),
            "C:/Users/Xx7/Desktop/yolov26s_15_lab_final/weights/best.pt",
        ),
        "```",
        "",
    ]
    (target / "README.md").write_text("\n".join(notes), encoding="utf-8")
    summary = {
        "session_dir": str(session),
        "output_dir": str(target),
        "ground_truth_path": str(ground_truth_path) if ground_truth_path else None,
        "candidate_source_path": str(source_rows_path) if source_rows_path else None,
        "capability_gap_path": str(capability_gap_path) if capability_gap_path else None,
        "eval_config_path": str(eval_config_path) if eval_config_path else None,
        "yolo_rows_path": str(yolo_rows_source),
        "views": selected_views,
        "target_classes": target_classes,
        "source_row_count": len(source_rows),
        "image_count": image_count,
        "prelabel_count": prelabel_count,
        "candidate_count": len(candidate_rows),
        "empty_template_count": empty_template_count,
        "failed_extract_count": len(failed_extracts),
        "failed_extracts": failed_extracts[:20],
        "dataset_yaml": str(target / "dataset.yaml"),
        "review_manifest": str(target / "review_manifest.jsonl"),
        "candidate_manifest": str(target / "candidate_manifest.jsonl"),
        "training_ready": False,
        "training_ready_reason": "prelabels and missing-class candidates require human bbox review before retraining",
        "confirmation_policy": "candidate rows cannot automatically upgrade confirmed evidence",
        "auto_confirmed_classes": False,
    }
    _write_json(target / "summary.json", summary)
    return summary


def _load_manifest_optional(path: Path) -> SessionManifest | None:
    if not path.exists():
        return None
    try:
        return SessionManifest.load(path)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _load_candidate_source_rows(
    session: Path,
    ground_truth_path: str | Path | None,
    candidate_source_path: str | Path | None,
) -> tuple[list[dict[str, Any]], Path | None]:
    if candidate_source_path:
        path = Path(candidate_source_path)
        return read_jsonl(path), path
    if ground_truth_path:
        path = Path(ground_truth_path)
        return read_jsonl(path), path
    for relative in _CANDIDATE_SOURCE_PATHS:
        path = session / relative
        if path.exists():
            return read_jsonl(path), path
    return [], None


def _target_missing_classes(
    capability_gap_path: str | Path | None,
    missing_classes: list[str] | None,
) -> list[str]:
    values: list[str] = []
    if capability_gap_path:
        path = Path(capability_gap_path)
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            values.extend(str(item) for item in report.get("recommended_new_classes") or [])
            targets = report.get("minimum_label_targets")
            if isinstance(targets, Mapping):
                values.extend(str(item) for item in targets.keys())
            plan = report.get("annotation_plan")
            if isinstance(plan, Mapping):
                for task in plan.get("capability_tasks") or []:
                    if isinstance(task, Mapping):
                        values.extend(str(item) for item in task.get("target_classes") or [])
    for item in missing_classes or []:
        values.extend(_expand_missing_class(item))
    return _merge_classes([], values)


def _expand_missing_class(value: Any) -> list[str]:
    name = canonical_yolo_label(value)
    requirement = CAPABILITY_REQUIREMENTS.get(name)
    if requirement:
        return [canonical_yolo_label(item) for item in requirement.get("recommended_new_classes") or []]
    if name == "liquid_stream":
        return ["liquid_stream"]
    if name == "meniscus":
        return ["meniscus_line", "liquid_level"]
    return [name] if name else []


def _merge_classes(base_classes: list[str], extra_classes: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for class_name in [*base_classes, *extra_classes]:
        canonical = canonical_yolo_label(class_name)
        if canonical and canonical not in seen:
            seen.add(canonical)
            merged.append(canonical)
    return merged


def _missing_class_candidate_seeds(
    source_rows: list[dict[str, Any]],
    target_classes: list[str],
    *,
    max_candidates_per_class: int,
) -> list[dict[str, Any]]:
    if not source_rows or not target_classes:
        return []
    limit = max(1, int(max_candidates_per_class))
    seeds: list[dict[str, Any]] = []
    for class_name in target_classes:
        scored = []
        for index, row in enumerate(source_rows, start=1):
            score = _candidate_score(class_name, row)
            scored.append((score, index, row))
        positive = [item for item in scored if item[0] > 0]
        selected = positive if positive else scored
        selected = sorted(selected, key=lambda item: (-item[0], item[1]))[:limit]
        for score, index, row in selected:
            seeds.append(
                {
                    "target_class": class_name,
                    "source_row_key": _source_row_key(row, index),
                    "source_row_id": _source_row_id(row, index),
                    "candidate_score": int(score),
                    "candidate_reason": "keyword_match" if score > 0 else "fallback_manual_scan",
                }
            )
    return seeds


def _source_row_key(row: Mapping[str, Any], index: int) -> str:
    return f"{_source_row_id(row, index)}#{index:05d}"


def _source_row_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("micro_segment_id", "segment_id", "event_id", "id", "source_prediction_id"):
        if row.get(key):
            return str(row[key])
    return f"row_{index:05d}"


def _candidate_score(class_name: str, row: Mapping[str, Any]) -> int:
    text = _row_search_text(row)
    if not text:
        return 0
    class_parts = [part for part in class_name.split("_") if len(part) >= 3]
    hints = _hints_for_class(class_name)
    score = 0
    if class_name in text:
        score += 5
    score += sum(2 for part in class_parts if part in text)
    score += sum(1 for hint in hints if hint in text)
    return score


def _hints_for_class(class_name: str) -> list[str]:
    hints: list[str] = []
    if any(token in class_name for token in ("liquid", "stream", "meniscus", "level")):
        hints.extend(_MISSING_CLASS_HINTS["liquid"])
    if any(token in class_name for token in ("button", "knob", "display", "readout")):
        hints.extend(_MISSING_CLASS_HINTS["control"])
    if any(token in class_name for token in ("open", "closed", "cap", "lid")):
        hints.extend(_MISSING_CLASS_HINTS["container_state"])
    return _merge_classes([], hints)


def _row_search_text(value: Any) -> str:
    parts: list[str] = []
    if isinstance(value, Mapping):
        for item in value.values():
            text = _row_search_text(item)
            if text:
                parts.append(text)
    elif isinstance(value, list):
        for item in value:
            text = _row_search_text(item)
            if text:
                parts.append(text)
    elif value is not None:
        parts.append(str(value))
    return canonical_yolo_label(" ".join(parts))

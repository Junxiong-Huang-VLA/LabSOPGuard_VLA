from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .model_inventory import (
    _canonical_class,
    _find_labsopguard_root,
    _matching_label_dir,
    _parse_class_schema,
    _parse_dataset_yaml,
    _resolve_dataset_root,
    _resolve_dataset_split,
    _resolve_project_root,
    discover_lab_assets,
    load_model_inventory,
)


CAPABILITY_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "liquid": {
        "aliases": ["liquid", "fluid", "solution"],
        "recommended_new_classes": ["liquid_region"],
        "inventory_capabilities": ["liquid_stream_segmentation"],
    },
    "stream": {
        "aliases": ["stream", "liquid_stream", "pour_stream"],
        "recommended_new_classes": ["liquid_stream"],
        "inventory_capabilities": ["liquid_stream_segmentation"],
    },
    "meniscus": {
        "aliases": ["meniscus", "liquid_level", "level"],
        "recommended_new_classes": ["meniscus_line", "liquid_level"],
        "inventory_capabilities": ["liquid_stream_segmentation"],
    },
    "button": {
        "aliases": ["button", "push_button"],
        "recommended_new_classes": ["button"],
        "inventory_capabilities": ["equipment_control_state_detection"],
    },
    "knob": {
        "aliases": ["knob", "dial"],
        "recommended_new_classes": ["knob"],
        "inventory_capabilities": ["equipment_control_state_detection"],
    },
    "display": {
        "aliases": ["display", "screen", "readout"],
        "recommended_new_classes": ["display", "display_readout"],
        "inventory_capabilities": ["equipment_control_state_detection"],
    },
    "open": {
        "aliases": ["open", "opened", "cap_open", "lid_open", "container_open"],
        "recommended_new_classes": ["container_open", "cap_open", "lid_open"],
        "inventory_capabilities": ["cap_lid_detection", "container_detection"],
    },
    "closed": {
        "aliases": ["closed", "cap_closed", "lid_closed", "container_closed"],
        "recommended_new_classes": ["container_closed", "cap_closed", "lid_closed"],
        "inventory_capabilities": ["cap_lid_detection", "container_detection"],
    },
}


_CAPABILITY_LABEL_TARGETS: dict[str, dict[str, Any]] = {
    "liquid": {
        "min_total_instances": 180,
        "min_positive_frames": 90,
        "min_video_sessions": 4,
        "min_views": 2,
        "min_val_instances": 36,
        "negative_context_frames": 90,
        "label_style": "bbox_or_mask_region",
        "must_cover": ["clear vessel boundary", "partially occluded liquid", "hand-object interaction context"],
    },
    "stream": {
        "min_total_instances": 220,
        "min_positive_frames": 120,
        "min_video_sessions": 4,
        "min_views": 2,
        "min_val_instances": 44,
        "negative_context_frames": 100,
        "label_style": "thin_bbox_or_mask_stream",
        "must_cover": ["pour stream", "pipette stream", "short transient flow", "false-positive empty background"],
    },
    "meniscus": {
        "min_total_instances": 180,
        "min_positive_frames": 90,
        "min_video_sessions": 4,
        "min_views": 2,
        "min_val_instances": 36,
        "negative_context_frames": 90,
        "label_style": "line_box_or_mask_level",
        "must_cover": ["tube meniscus", "beaker liquid level", "occluded or reflective surface"],
    },
    "button": {
        "min_total_instances": 160,
        "min_positive_frames": 80,
        "min_video_sessions": 3,
        "min_views": 2,
        "min_val_instances": 32,
        "negative_context_frames": 80,
        "label_style": "bbox_control_part",
        "must_cover": ["pressed and unpressed appearances", "operator hand near control", "panel background negatives"],
    },
    "knob": {
        "min_total_instances": 160,
        "min_positive_frames": 80,
        "min_video_sessions": 3,
        "min_views": 2,
        "min_val_instances": 32,
        "negative_context_frames": 80,
        "label_style": "bbox_control_part",
        "must_cover": ["different dial angles", "operator hand near control", "panel background negatives"],
    },
    "display": {
        "min_total_instances": 180,
        "min_positive_frames": 90,
        "min_video_sessions": 3,
        "min_views": 2,
        "min_val_instances": 36,
        "negative_context_frames": 80,
        "label_style": "bbox_display_or_readout",
        "must_cover": ["readable display", "glare or partial occlusion", "device panel negatives"],
    },
    "open": {
        "min_total_instances": 160,
        "min_positive_frames": 80,
        "min_video_sessions": 4,
        "min_views": 2,
        "min_val_instances": 32,
        "negative_context_frames": 80,
        "label_style": "bbox_state_specific_container_or_lid",
        "must_cover": ["cap removed", "lid open", "hand near closure", "closed negative examples"],
    },
    "closed": {
        "min_total_instances": 160,
        "min_positive_frames": 80,
        "min_video_sessions": 4,
        "min_views": 2,
        "min_val_instances": 32,
        "negative_context_frames": 80,
        "label_style": "bbox_state_specific_container_or_lid",
        "must_cover": ["cap attached", "lid closed", "hand near closure", "open negative examples"],
    },
}

_ANNOTATION_CANDIDATE_SOURCES = [
    "metadata/micro_segments.jsonl",
    "metadata/key_action_segments.jsonl",
    "metadata/vector_metadata.jsonl",
    "cv_outputs/yolo_micro_frame_rows.jsonl",
    "metadata/advanced_vision_evidence.jsonl",
    "metadata/model_observation_events.jsonl",
]


def build_capability_gap_report(
    project_root: str | Path | None = None,
    *,
    model_inventory_path: str | Path | None = None,
    model_inventory: Mapping[str, Any] | None = None,
    dataset_root: str | Path | None = None,
    class_schema_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    inventory = dict(model_inventory or {})
    inventory_path = Path(model_inventory_path).resolve() if model_inventory_path else None
    if not inventory and inventory_path:
        inventory = load_model_inventory(inventory_path)

    root_hint = project_root or inventory.get("project_root")
    root = _resolve_project_root(root_hint)
    if not inventory:
        inventory = discover_lab_assets(project_root=root)

    labsopguard_root = _labsopguard_root(root, inventory)
    schema_path = _class_schema_path(class_schema_path, labsopguard_root, inventory)
    class_sources = _collect_class_sources(inventory, schema_path)
    dataset_specs = _collect_dataset_specs(
        root=root,
        labsopguard_root=labsopguard_root,
        inventory=inventory,
        dataset_root=dataset_root,
    )

    for spec in dataset_specs:
        for class_name in spec.get("class_names") or []:
            _add_class_source(class_sources, class_name, "dataset_yaml")

    label_stats = _scan_yolo_labels(dataset_specs, class_sources)
    available_classes = sorted(class_sources)
    capability_rows = _capability_rows(
        available_classes=available_classes,
        class_sources=class_sources,
        label_sample_counts=label_stats["class_counts"],
        label_file_counts=label_stats["class_file_counts"],
        split_counts=label_stats["split_counts"],
        inventory=inventory,
    )
    missing_classes = sorted(
        {
            class_name
            for capability in capability_rows.values()
            if not capability["has_label_foundation"]
            for class_name in capability["recommended_new_classes"]
            if class_name not in available_classes
        }
    )
    recommended_new_classes = sorted(
        {
            class_name
            for capability in capability_rows.values()
            if not capability["has_label_foundation"]
            for class_name in capability["recommended_new_classes"]
        }
    )
    minimum_label_targets = _minimum_label_targets(capability_rows)
    annotation_plan = _annotation_plan(
        capability_rows=capability_rows,
        minimum_label_targets=minimum_label_targets,
        recommended_new_classes=recommended_new_classes,
    )
    blocking_unavailable_capabilities = _blocking_unavailable_capabilities(capability_rows)
    report = {
        "metadata_version": "key_action_capability_gap_report.v1",
        "project_root": str(root),
        "labsopguard_root": str(labsopguard_root) if labsopguard_root else None,
        "source_paths": {
            "model_inventory": str(inventory_path) if inventory_path else None,
            "class_schema": str(schema_path) if schema_path and schema_path.exists() else None,
            "datasets": [str(spec.get("path") or spec.get("dataset_root")) for spec in dataset_specs],
        },
        "summary": _summary(inventory, available_classes, capability_rows, label_stats, dataset_specs),
        "available_classes": available_classes,
        "class_details": _class_details(class_sources, label_stats),
        "label_sample_counts": _label_sample_counts(label_stats),
        "capabilities": capability_rows,
        "missing_classes": missing_classes,
        "recommended_new_classes": recommended_new_classes,
        "audit_findings": _audit_findings(capability_rows, inventory),
        "dataset_audit": _dataset_audit(dataset_specs, label_stats),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _labsopguard_root(root: Path, inventory: Mapping[str, Any]) -> Path | None:
    value = inventory.get("labsopguard_root")
    if value:
        path = Path(str(value))
        if path.exists():
            return path.resolve()
    return _find_labsopguard_root(root)


def _class_schema_path(
    explicit_path: str | Path | None,
    labsopguard_root: Path | None,
    inventory: Mapping[str, Any],
) -> Path | None:
    if explicit_path:
        return Path(explicit_path).resolve()
    value = inventory.get("class_schema_path")
    if value:
        path = Path(str(value))
        if path.exists():
            return path.resolve()
    if labsopguard_root:
        return labsopguard_root / "configs" / "data" / "class_schema.yaml"
    return None


def _collect_class_sources(inventory: Mapping[str, Any], class_schema_path: Path | None) -> dict[str, set[str]]:
    sources: dict[str, set[str]] = {}
    for class_name in inventory.get("classes") or []:
        _add_class_source(sources, class_name, "model_inventory")
    schema = inventory.get("class_schema") if isinstance(inventory.get("class_schema"), Mapping) else {}
    for item in schema.get("classes") or []:
        if isinstance(item, Mapping) and item.get("name"):
            _add_class_source(sources, item["name"], "class_schema")
    if class_schema_path and class_schema_path.exists():
        parsed = _parse_class_schema(class_schema_path)
        for item in parsed.get("classes") or []:
            if isinstance(item, Mapping) and item.get("name"):
                _add_class_source(sources, item["name"], "class_schema")
    return sources


def _add_class_source(sources: dict[str, set[str]], value: Any, source: str) -> None:
    class_name = _canonical_class(value)
    if class_name:
        sources.setdefault(class_name, set()).add(source)


def _collect_dataset_specs(
    *,
    root: Path,
    labsopguard_root: Path | None,
    inventory: Mapping[str, Any],
    dataset_root: str | Path | None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in inventory.get("datasets") or []:
        if isinstance(item, Mapping):
            spec = _dataset_spec_from_inventory(item)
            if spec:
                specs.append(spec)

    if dataset_root:
        spec = _dataset_spec_from_root(Path(dataset_root), labsopguard_root)
        if spec:
            specs.append(spec)
    elif labsopguard_root:
        spec = _dataset_spec_from_root(labsopguard_root / "data" / "dataset", labsopguard_root)
        if spec:
            specs.append(spec)

    return _dedupe_dataset_specs(specs)


def _dataset_spec_from_inventory(item: Mapping[str, Any]) -> dict[str, Any] | None:
    yaml_path = Path(str(item.get("path") or ""))
    class_names = [_canonical_class(name) for name in item.get("class_names") or []]
    if yaml_path.exists() and not class_names:
        class_names = [_canonical_class(name) for name in _parse_dataset_yaml(yaml_path).get("names") or []]
    label_specs = []
    splits = item.get("splits") if isinstance(item.get("splits"), Mapping) else {}
    for split, split_info in splits.items():
        if not isinstance(split_info, Mapping):
            continue
        image_path = split_info.get("path")
        if not image_path:
            continue
        label_dir = _matching_label_dir(Path(str(image_path)))
        if label_dir and label_dir.exists():
            label_specs.append({"split": str(split), "label_dir": label_dir, "class_names": class_names})
    dataset_root = Path(str(item.get("dataset_root") or "")) if item.get("dataset_root") else None
    if dataset_root and dataset_root.exists() and not label_specs:
        label_specs.extend(_label_specs_from_dataset_root(dataset_root, class_names))
    if not label_specs and not class_names and (not yaml_path or not yaml_path.exists()):
        return None
    return {
        "path": str(yaml_path) if yaml_path else None,
        "dataset_root": str(dataset_root) if dataset_root else item.get("dataset_root"),
        "class_names": class_names,
        "label_specs": label_specs,
    }


def _dataset_spec_from_root(dataset_root: Path, labsopguard_root: Path | None) -> dict[str, Any] | None:
    root = dataset_root.resolve()
    yaml_path = root / "dataset.yaml" if root.is_dir() else root
    if not yaml_path.exists():
        labels_dir = root / "labels" if root.is_dir() else None
        if labels_dir and labels_dir.exists():
            return {
                "path": None,
                "dataset_root": str(root),
                "class_names": [],
                "label_specs": _label_specs_from_dataset_root(root, []),
            }
        return None
    data = _parse_dataset_yaml(yaml_path)
    resolved_root = _resolve_dataset_root(labsopguard_root or yaml_path.parent, yaml_path, data.get("path")) or yaml_path.parent
    class_names = [_canonical_class(name) for name in data.get("names") or []]
    label_specs = []
    for split in ("train", "val", "test"):
        image_dir = _resolve_dataset_split(resolved_root, yaml_path, data.get(split))
        label_dir = _matching_label_dir(image_dir)
        if label_dir and label_dir.exists():
            label_specs.append({"split": split, "label_dir": label_dir, "class_names": class_names})
    if not label_specs:
        label_specs.extend(_label_specs_from_dataset_root(resolved_root, class_names))
    return {
        "path": str(yaml_path),
        "dataset_root": str(resolved_root),
        "class_names": class_names,
        "label_specs": label_specs,
    }


def _label_specs_from_dataset_root(dataset_root: Path, class_names: list[str]) -> list[dict[str, Any]]:
    labels_dir = dataset_root / "labels"
    if not labels_dir.exists():
        return []
    specs = []
    for split in ("train", "val", "test"):
        split_dir = labels_dir / split
        if split_dir.exists():
            specs.append({"split": split, "label_dir": split_dir, "class_names": class_names})
    if not specs:
        specs.append({"split": "all", "label_dir": labels_dir, "class_names": class_names})
    return specs


def _dedupe_dataset_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for spec in specs:
        label_dirs = tuple(sorted(str(Path(str(item["label_dir"])).resolve()) for item in spec.get("label_specs") or []))
        key = (str(spec.get("path") or spec.get("dataset_root") or ""), label_dirs)
        if key in seen:
            continue
        seen.add(key)
        result.append(spec)
    return result


def _scan_yolo_labels(dataset_specs: list[dict[str, Any]], class_sources: dict[str, set[str]]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_files: dict[str, set[str]] = defaultdict(set)
    label_files_by_split: Counter[str] = Counter()
    empty_label_files = 0
    invalid_lines = 0
    seen_files: set[Path] = set()
    scanned_dirs: list[str] = []

    for spec in dataset_specs:
        for label_spec in spec.get("label_specs") or []:
            label_dir = Path(str(label_spec.get("label_dir") or ""))
            if not label_dir.exists():
                continue
            class_names = list(label_spec.get("class_names") or spec.get("class_names") or [])
            split = str(label_spec.get("split") or "unknown")
            scanned_dirs.append(str(label_dir))
            for label_file in sorted(label_dir.rglob("*.txt")):
                resolved = label_file.resolve()
                if resolved in seen_files:
                    continue
                seen_files.add(resolved)
                label_files_by_split[split] += 1
                valid_in_file = 0
                try:
                    lines = label_file.read_text(encoding="utf-8-sig").splitlines()
                except OSError:
                    invalid_lines += 1
                    continue
                for line in lines:
                    class_name = _class_from_yolo_line(line, class_names)
                    if not class_name:
                        if line.strip():
                            invalid_lines += 1
                        continue
                    valid_in_file += 1
                    class_counts[class_name] += 1
                    split_counts[split][class_name] += 1
                    class_files[class_name].add(str(resolved))
                    class_sources.setdefault(class_name, set()).add("yolo_labels")
                if valid_in_file == 0:
                    empty_label_files += 1

    return {
        "class_counts": class_counts,
        "split_counts": split_counts,
        "class_file_counts": {class_name: len(paths) for class_name, paths in class_files.items()},
        "label_file_count": len(seen_files),
        "label_files_by_split": dict(label_files_by_split),
        "empty_label_file_count": empty_label_files,
        "invalid_label_line_count": invalid_lines,
        "scanned_label_dirs": sorted(set(scanned_dirs)),
    }


def _class_from_yolo_line(line: str, class_names: list[str]) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        class_id = int(float(parts[0]))
    except ValueError:
        return None
    if class_id < 0:
        return None
    if class_id < len(class_names) and class_names[class_id]:
        return _canonical_class(class_names[class_id])
    return f"class_{class_id}"


def _capability_rows(
    *,
    available_classes: list[str],
    class_sources: Mapping[str, set[str]],
    label_sample_counts: Mapping[str, int],
    label_file_counts: Mapping[str, int],
    split_counts: Mapping[str, Counter[str]],
    inventory: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for capability, requirement in CAPABILITY_REQUIREMENTS.items():
        aliases = [_canonical_class(item) for item in requirement["aliases"]]
        matching_classes = [class_name for class_name in available_classes if _matches_any_alias(class_name, aliases)]
        sample_count = sum(int(label_sample_counts.get(class_name, 0)) for class_name in matching_classes)
        labeled_file_count = sum(int(label_file_counts.get(class_name, 0)) for class_name in matching_classes)
        by_split = {
            split: sum(int(counts.get(class_name, 0)) for class_name in matching_classes)
            for split, counts in split_counts.items()
        }
        by_split = {split: count for split, count in sorted(by_split.items()) if count}
        has_class_definition = bool(matching_classes)
        has_label_foundation = sample_count > 0
        rows[capability] = {
            "aliases": aliases,
            "required_terms": aliases,
            "matching_classes": matching_classes,
            "class_sources": sorted({source for class_name in matching_classes for source in class_sources.get(class_name, set())}),
            "has_class_definition": has_class_definition,
            "has_label_foundation": has_label_foundation,
            "sample_count": sample_count,
            "labeled_file_count": labeled_file_count,
            "sample_counts_by_split": by_split,
            "status": _capability_status(has_class_definition, has_label_foundation),
            "recommended_new_classes": [
                _canonical_class(item)
                for item in requirement["recommended_new_classes"]
                if _canonical_class(item) not in matching_classes
            ],
            "inventory_capability_refs": _inventory_capability_refs(inventory, requirement.get("inventory_capabilities") or []),
        }
    return rows


def _minimum_label_targets(capability_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for capability_name, row in sorted(capability_rows.items()):
        if row.get("has_label_foundation"):
            continue
        target_classes = list(row.get("recommended_new_classes") or row.get("matching_classes") or [])
        base = dict(_CAPABILITY_LABEL_TARGETS.get(capability_name, {}))
        if not base:
            base = {
                "min_total_instances": 160,
                "min_positive_frames": 80,
                "min_video_sessions": 3,
                "min_views": 2,
                "min_val_instances": 32,
                "negative_context_frames": 80,
                "label_style": "bbox",
                "must_cover": ["first-person view", "third-person view", "negative context"],
            }
        for class_name in target_classes:
            class_targets = dict(base)
            class_targets["capability"] = capability_name
            class_targets["class_name"] = class_name
            class_targets["manual_review_required"] = True
            class_targets["training_gate"] = "do not train or mark confirmed until these reviewed labels exist"
            targets[class_name] = class_targets
    return targets


def _annotation_plan(
    *,
    capability_rows: Mapping[str, Mapping[str, Any]],
    minimum_label_targets: Mapping[str, Mapping[str, Any]],
    recommended_new_classes: list[str],
) -> dict[str, Any]:
    tasks = []
    for capability_name, row in sorted(capability_rows.items()):
        if row.get("has_label_foundation"):
            continue
        target_classes = sorted(row.get("recommended_new_classes") or row.get("matching_classes") or [])
        tasks.append(
            {
                "capability": capability_name,
                "status": row.get("status"),
                "priority": _annotation_priority(capability_name),
                "target_classes": target_classes,
                "existing_matching_classes": row.get("matching_classes", []),
                "class_definitions_to_add": sorted(row.get("recommended_new_classes") or []),
                "minimum_label_targets": {
                    class_name: minimum_label_targets[class_name]
                    for class_name in target_classes
                    if class_name in minimum_label_targets
                },
                "candidate_sources": list(_ANNOTATION_CANDIDATE_SOURCES),
                "human_review_required": True,
                "cannot_auto_upgrade_confirmed": True,
                "review_output_required": [
                    "reviewed YOLO labels under labels/train and labels/val",
                    "candidate manifest with accepted/rejected status",
                    "updated dataset.yaml containing the new class ids",
                ],
                "retrieval_use": "feed reviewed detections into micro-segment metadata and vector retrieval evidence",
            }
        )
    return {
        "status": "needs_annotation" if tasks else "label_foundation_present",
        "recommended_new_classes": recommended_new_classes,
        "candidate_sources": list(_ANNOTATION_CANDIDATE_SOURCES),
        "capability_tasks": tasks,
        "training_gate": "missing classes are candidate-only until human bbox review and retraining are complete",
        "confirmation_policy": "candidate labels from relabel packs must not automatically upgrade process evidence to confirmed",
    }


def _annotation_priority(capability_name: str) -> str:
    if capability_name in {"stream", "meniscus", "liquid"}:
        return "P0"
    if capability_name in {"open", "closed", "button", "knob", "display"}:
        return "P1"
    return "P2"


def _blocking_unavailable_capabilities(capability_rows: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    blocked: dict[tuple[str, str], dict[str, Any]] = {}
    for capability_name, row in capability_rows.items():
        if row.get("has_label_foundation"):
            continue
        refs = row.get("inventory_capability_refs") if isinstance(row.get("inventory_capability_refs"), Mapping) else {}
        for inventory_capability, info in refs.items():
            if isinstance(info, Mapping) and info.get("available"):
                continue
            key = (str(inventory_capability), str(capability_name))
            blocked[key] = {
                "inventory_capability": str(inventory_capability),
                "blocks_required_capability": str(capability_name),
                "reason": "model inventory has no available class/model coverage for this capability",
                "recommended_new_classes": sorted(row.get("recommended_new_classes") or []),
                "minimum_label_targets": sorted(row.get("recommended_new_classes") or row.get("matching_classes") or []),
            }
    return [blocked[key] for key in sorted(blocked)]


def _suggested_commands(
    *,
    root: Path,
    inventory_path: Path | None,
    dataset_root: str | Path | None,
    class_schema_path: Path | None,
    output_path: str | Path | None,
    recommended_new_classes: list[str],
) -> list[dict[str, str]]:
    report_path = Path(output_path) if output_path else Path("data/sessions/<session_id>/metadata/capability_gap_report.json")
    gap_parts = [
        "python -m key_action_indexer.cli capability-gap-report",
        "--project-root",
        _quote_cli(root),
    ]
    if inventory_path:
        gap_parts.extend(["--model-inventory", _quote_cli(inventory_path)])
    if dataset_root:
        gap_parts.extend(["--dataset-root", _quote_cli(dataset_root)])
    if class_schema_path:
        gap_parts.extend(["--class-schema", _quote_cli(class_schema_path)])
    if output_path:
        gap_parts.extend(["--output", _quote_cli(output_path)])

    relabel_parts = [
        "python -m key_action_indexer.cli export-yolo-relabel-pack",
        "--session-dir <session_dir>",
        f"--capability-gap {_quote_cli(report_path)}",
        "--ground-truth <micro_ground_truth.jsonl>",
    ]
    relabel_parts.extend(f"--missing-class {class_name}" for class_name in recommended_new_classes)
    relabel_command = " ".join(relabel_parts)
    return [
        {
            "step": "refresh_gap_report",
            "command": " ".join(str(part) for part in gap_parts),
        },
        {
            "step": "generate_missing_class_relabel_pack",
            "command": relabel_command,
        },
        {
            "step": "train_after_human_review",
            "command": (
                "python LabSOPGuard/scripts/train_yolo_lab.py "
                "--dataset-yaml <reviewed_relabel_pack/dataset.yaml> "
                "--model <current_best.pt> --epochs 50 --imgsz 960 --batch 8"
            ),
        },
        {
            "step": "rerun_key_action_metadata_and_retrieval",
            "command": "python -m key_action_indexer.cli run --manifest <manifest.json>",
        },
    ]


def _quote_cli(value: str | Path) -> str:
    text = str(value)
    if any(char.isspace() for char in text):
        return f'"{text}"'
    return text


def _matches_any_alias(class_name: str, aliases: Iterable[str]) -> bool:
    parts = set(class_name.split("_"))
    for alias in aliases:
        if class_name == alias or alias in parts:
            return True
        if "_" in alias and alias in class_name:
            return True
        if alias in {"liquid", "fluid", "solution", "stream", "meniscus", "button", "knob", "display", "screen", "readout", "open", "closed"}:
            if alias in class_name:
                return True
    return False


def _capability_status(has_class_definition: bool, has_label_foundation: bool) -> str:
    if has_label_foundation:
        return "labeled"
    if has_class_definition:
        return "schema_without_samples"
    return "missing_class"


def _inventory_capability_refs(inventory: Mapping[str, Any], names: Iterable[str]) -> dict[str, Any]:
    capabilities = inventory.get("capabilities") if isinstance(inventory.get("capabilities"), Mapping) else {}
    refs = {}
    for name in names:
        item = capabilities.get(name)
        if isinstance(item, Mapping):
            refs[name] = {
                "available": bool(item.get("available")),
                "classes": item.get("classes", []),
                "model_count": item.get("model_count"),
            }
        else:
            refs[name] = {"available": False, "classes": [], "model_count": None}
    return refs


def _summary(
    inventory: Mapping[str, Any],
    available_classes: list[str],
    capability_rows: Mapping[str, Mapping[str, Any]],
    label_stats: Mapping[str, Any],
    dataset_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    missing = [name for name, row in capability_rows.items() if not row.get("has_label_foundation")]
    schema_only = [name for name, row in capability_rows.items() if row.get("has_class_definition") and not row.get("has_label_foundation")]
    labeled = [name for name, row in capability_rows.items() if row.get("has_label_foundation")]
    inventory_caps = inventory.get("capabilities") if isinstance(inventory.get("capabilities"), Mapping) else {}
    unavailable_inventory = [
        name
        for name, item in inventory_caps.items()
        if isinstance(item, Mapping) and not item.get("available")
    ]
    return {
        "status": "needs_annotation" if missing else "label_foundation_present",
        "required_capability_count": len(capability_rows),
        "capabilities_with_label_foundation": labeled,
        "capabilities_missing_label_foundation": missing,
        "schema_only_capabilities": schema_only,
        "available_class_count": len(available_classes),
        "dataset_count": len(dataset_specs),
        "model_count": int(inventory.get("model_count") or 0),
        "total_yolo_label_files": int(label_stats.get("label_file_count") or 0),
        "total_yolo_label_instances": int(sum(label_stats.get("class_counts", Counter()).values())),
        "empty_yolo_label_files": int(label_stats.get("empty_label_file_count") or 0),
        "invalid_yolo_label_lines": int(label_stats.get("invalid_label_line_count") or 0),
        "unavailable_inventory_capabilities": sorted(unavailable_inventory),
    }


def _class_details(class_sources: Mapping[str, set[str]], label_stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    counts = label_stats.get("class_counts", Counter())
    file_counts = label_stats.get("class_file_counts", {})
    return [
        {
            "class_name": class_name,
            "sources": sorted(sources),
            "sample_count": int(counts.get(class_name, 0)),
            "labeled_file_count": int(file_counts.get(class_name, 0)),
        }
        for class_name, sources in sorted(class_sources.items())
    ]


def _label_sample_counts(label_stats: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    counts = label_stats.get("class_counts", Counter())
    file_counts = label_stats.get("class_file_counts", {})
    split_counts = label_stats.get("split_counts", {})
    result = {}
    for class_name in sorted(counts):
        result[class_name] = {
            "instances": int(counts.get(class_name, 0)),
            "labeled_files": int(file_counts.get(class_name, 0)),
            "splits": {
                split: int(values.get(class_name, 0))
                for split, values in sorted(split_counts.items())
                if int(values.get(class_name, 0)) > 0
            },
        }
    return result


def _audit_findings(capability_rows: Mapping[str, Mapping[str, Any]], inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    findings = []
    for name, row in capability_rows.items():
        if row.get("has_label_foundation"):
            continue
        reason = "class exists but YOLO labels contain no samples" if row.get("has_class_definition") else "class is absent from schema and dataset labels"
        findings.append(
            {
                "capability": name,
                "severity": "gap",
                "reason": reason,
                "recommend_new_classes": row.get("recommended_new_classes", []),
            }
        )
    capabilities = inventory.get("capabilities") if isinstance(inventory.get("capabilities"), Mapping) else {}
    for capability_name in ("liquid_stream_segmentation", "equipment_control_state_detection"):
        item = capabilities.get(capability_name)
        if isinstance(item, Mapping) and not item.get("available"):
            findings.append(
                {
                    "capability": capability_name,
                    "severity": "gap",
                    "reason": "model inventory reports this capability as unavailable",
                    "recommend_new_classes": [],
                }
            )
    return findings


def _dataset_audit(dataset_specs: list[dict[str, Any]], label_stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_count": len(dataset_specs),
        "scanned_label_dirs": label_stats.get("scanned_label_dirs", []),
        "label_files_by_split": label_stats.get("label_files_by_split", {}),
        "empty_label_file_count": int(label_stats.get("empty_label_file_count") or 0),
        "invalid_label_line_count": int(label_stats.get("invalid_label_line_count") or 0),
        "datasets": [
            {
                "path": spec.get("path"),
                "dataset_root": spec.get("dataset_root"),
                "class_count": len(spec.get("class_names") or []),
                "label_dirs": [str(item.get("label_dir")) for item in spec.get("label_specs") or []],
            }
            for spec in dataset_specs
        ],
    }


__all__ = ["CAPABILITY_REQUIREMENTS", "build_capability_gap_report"]

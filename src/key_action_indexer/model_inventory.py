from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MODEL_EXTENSIONS = {".pt", ".onnx", ".engine", ".pth", ".safetensors"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTENSIONS = {".txt", ".json", ".jsonl", ".xml"}


def discover_lab_assets(project_root: str | Path | None = None, output_path: str | Path | None = None) -> dict[str, Any]:
    root = _resolve_project_root(project_root)
    labsopguard_root = _find_labsopguard_root(root)
    runtime_config_path = labsopguard_root / "configs" / "model" / "detection_runtime.yaml" if labsopguard_root else None
    class_schema_path = labsopguard_root / "configs" / "data" / "class_schema.yaml" if labsopguard_root else None
    pose_schema_path = labsopguard_root / "configs" / "data" / "pose_keypoints_schema.yaml" if labsopguard_root else None
    runtime_config = _parse_runtime_config(runtime_config_path)
    class_schema = _parse_class_schema(class_schema_path)
    pose_schema = _parse_pose_schema(pose_schema_path)
    models = _discover_models(root, labsopguard_root, runtime_config)
    datasets = _discover_datasets(labsopguard_root)
    classes = _canonical_classes(class_schema, datasets, runtime_config)
    capabilities = _capabilities(classes, models, pose_schema)
    inventory = {
        "metadata_version": "key_action_model_inventory.v1",
        "project_root": str(root),
        "labsopguard_root": str(labsopguard_root) if labsopguard_root else None,
        "runtime_config_path": str(runtime_config_path) if runtime_config_path and runtime_config_path.exists() else None,
        "runtime_config": runtime_config,
        "class_schema_path": str(class_schema_path) if class_schema_path and class_schema_path.exists() else None,
        "class_schema": class_schema,
        "pose_schema_path": str(pose_schema_path) if pose_schema_path and pose_schema_path.exists() else None,
        "pose_schema": pose_schema,
        "primary_model": models[0] if models else None,
        "model_count": len(models),
        "models": models,
        "dataset_count": len(datasets),
        "datasets": datasets,
        "classes": classes,
        "capabilities": capabilities,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    return inventory


def load_model_inventory(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8-sig"))


def resolve_best_model_path(explicit_path: str | Path | None = None, project_root: str | Path | None = None) -> Path | None:
    if explicit_path:
        resolved = _resolve_candidate_path(explicit_path, _resolve_project_root(project_root))
        if resolved and resolved.exists() and resolved.is_file():
            return resolved
        if resolved and resolved.exists() and resolved.is_dir():
            nested = _first_existing(resolved / name for name in ("weights/best.pt", "best.pt", "weights/best.onnx", "best.onnx"))
            if nested:
                return nested.resolve()
    inventory = discover_lab_assets(project_root=project_root)
    primary = inventory.get("primary_model") if isinstance(inventory.get("primary_model"), dict) else {}
    path = Path(str(primary.get("path") or "")) if primary.get("path") else None
    return path if path and path.exists() else None


def _resolve_project_root(project_root: str | Path | None = None) -> Path:
    if project_root:
        return Path(project_root).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "key_action_indexer").exists():
            return candidate
    return cwd


def _find_labsopguard_root(root: Path) -> Path | None:
    if (root / "configs").exists() and (root / "outputs").exists():
        return root.resolve()
    return None


def _discover_models(root: Path, labsopguard_root: Path | None, runtime_config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[tuple[Path, str]] = []
    for env_name in ("KEY_ACTION_YOLO_MODEL", "LABSOPGUARD_YOLO_MODEL"):
        env_value = os.environ.get(env_name)
        if env_value:
            path = _resolve_candidate_path(env_value, root)
            if path:
                candidates.append((path, f"env:{env_name}"))
    for key, role in (("model", "runtime_primary"), ("model_fallbacks", "runtime_fallback")):
        value = runtime_config.get(key)
        if isinstance(value, list):
            for item in value:
                path = _resolve_candidate_path(str(item), labsopguard_root or root)
                if path:
                    candidates.append((path, role))
        elif value:
            path = _resolve_candidate_path(str(value), labsopguard_root or root)
            if path:
                candidates.append((path, role))
    search_roots = [root]
    if labsopguard_root and labsopguard_root not in search_roots:
        search_roots.append(labsopguard_root)
    for search_root in search_roots:
        for base in (search_root / "outputs" / "training", search_root):
            if not base.exists():
                continue
            try:
                for path in base.rglob("*"):
                    if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
                        candidates.append((path, "discovered"))
            except OSError:
                continue
    unique: dict[str, dict[str, Any]] = {}
    for path, role in candidates:
        resolved = path.resolve()
        if resolved.is_dir():
            nested = _first_existing(resolved / name for name in ("weights/best.pt", "best.pt", "weights/best.onnx", "best.onnx"))
            if not nested:
                continue
            resolved = nested.resolve()
        if not resolved.exists() or not resolved.is_file() or resolved.suffix.lower() not in MODEL_EXTENSIONS:
            continue
        key = str(resolved)
        item = unique.get(key)
        if item is None:
            item = _model_item(resolved, role)
            unique[key] = item
        else:
            roles = set(item.get("roles") or [])
            roles.add(role)
            item["roles"] = sorted(roles)
            item["priority"] = max(float(item.get("priority") or 0.0), _model_priority(resolved, role))
    return sorted(unique.values(), key=lambda item: (-float(item.get("priority") or 0.0), str(item.get("path") or "")))


def _model_item(path: Path, role: str) -> dict[str, Any]:
    try:
        stat = path.stat()
        size_bytes = int(stat.st_size)
        modified = stat.st_mtime
    except OSError:
        size_bytes = 0
        modified = 0.0
    name = str(path).replace("\\", "/").lower()
    task = "pose" if "pose" in name else "detect"
    precision = "tensorrt" if path.suffix.lower() == ".engine" else ("onnx" if path.suffix.lower() == ".onnx" else "pytorch")
    return {
        "path": str(path),
        "name": path.parent.parent.name if path.parent.name == "weights" else path.stem,
        "file": path.name,
        "format": precision,
        "task": task,
        "roles": [role],
        "size_bytes": size_bytes,
        "modified_timestamp": round(float(modified), 3),
        "priority": _model_priority(path, role),
    }


def _model_priority(path: Path, role: str) -> float:
    name = str(path).replace("\\", "/").lower()
    score = 0.0
    if role.startswith("env"):
        score += 120.0
    if role == "runtime_primary":
        score += 100.0
    if role == "runtime_fallback":
        score += 70.0
    if "v4_focus_auto" in name:
        score += 20.0
    if "stage2" in name:
        score += 8.0
    if "lab_relabel" in name:
        score += 7.0
    if "best.pt" in name:
        score += 6.0
    if "best.onnx" in name:
        score += 4.0
    if path.suffix.lower() == ".engine":
        score += 3.0
    if "yolo26s" in name:
        score += 2.0
    if "yolo26n" in name:
        score += 1.0
    return score


def _discover_datasets(labsopguard_root: Path | None) -> list[dict[str, Any]]:
    if not labsopguard_root:
        return []
    candidates: list[Path] = []
    for relative in (
        "data/dataset/dataset.yaml",
        "configs/data/dataset.yaml",
    ):
        path = labsopguard_root / relative
        if path.exists():
            candidates.append(path)
    processed = labsopguard_root / "data" / "processed"
    if processed.exists():
        candidates.extend(path for path in processed.rglob("dataset.yaml") if path.is_file())
    unique: dict[str, dict[str, Any]] = {}
    for path in candidates:
        resolved = path.resolve()
        unique[str(resolved)] = _dataset_item(labsopguard_root, resolved)
    return sorted(unique.values(), key=lambda item: str(item.get("path") or ""))


def _dataset_item(labsopguard_root: Path, path: Path) -> dict[str, Any]:
    data = _parse_dataset_yaml(path)
    dataset_root = _resolve_dataset_root(labsopguard_root, path, data.get("path"))
    splits = {}
    for split in ("train", "val", "test"):
        split_path = _resolve_dataset_split(dataset_root, path, data.get(split))
        image_count = _count_files(split_path, IMAGE_EXTENSIONS)
        label_path = _matching_label_dir(split_path)
        label_count = _count_files(label_path, {".txt"}) if label_path else 0
        splits[split] = {
            "path": str(split_path) if split_path else None,
            "image_count": image_count,
            "label_count": label_count,
        }
    return {
        "path": str(path),
        "dataset_root": str(dataset_root) if dataset_root else None,
        "class_names": data.get("names", []),
        "class_count": len(data.get("names", [])),
        "splits": splits,
        "total_image_count": sum(int(item["image_count"]) for item in splits.values()),
        "total_label_count": sum(int(item["label_count"]) for item in splits.values()),
    }


def _parse_runtime_config(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    result: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not raw_line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key
            if value:
                result[key] = _yaml_value(value)
            else:
                result.setdefault(key, [])
            continue
        if current_key and line.strip().startswith("-"):
            item = line.strip()[1:].strip()
            result.setdefault(current_key, [])
            if isinstance(result[current_key], list):
                result[current_key].append(_yaml_value(item))
    return result


def _parse_dataset_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    names: dict[int, str] = {}
    in_names = False
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        clean = raw_line.split("#", 1)[0].rstrip()
        if not clean:
            continue
        stripped = clean.strip()
        if stripped == "names:":
            in_names = True
            continue
        if in_names:
            if not raw_line.startswith(" ") and ":" in stripped:
                in_names = False
            else:
                if ":" in stripped:
                    key, value = stripped.split(":", 1)
                    try:
                        names[int(key.strip())] = str(_yaml_value(value.strip()))
                    except ValueError:
                        pass
                continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            data[key.strip()] = _yaml_value(value.strip())
    data["names"] = [names[index] for index in sorted(names)]
    return data


def _parse_class_schema(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"classes": []}
    classes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line.startswith("- id:"):
            if current:
                classes.append(current)
            current = {"id": int(line.split(":", 1)[1].strip())}
        elif current is not None and line.startswith("name:"):
            current["name"] = str(_yaml_value(line.split(":", 1)[1].strip()))
        elif current is not None and line.startswith("description:"):
            current["description"] = str(_yaml_value(line.split(":", 1)[1].strip()))
    if current:
        classes.append(current)
    return {"classes": classes, "class_count": len(classes)}


def _parse_pose_schema(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    default_keypoints = None
    keypoint_names: dict[str, list[str]] = {}
    class_alias: dict[str, list[str]] = {}
    section = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("default_keypoints:"):
            try:
                default_keypoints = int(line.split(":", 1)[1].strip())
            except ValueError:
                default_keypoints = None
        elif line in {"keypoint_names:", "class_alias:"}:
            section = line.rstrip(":")
        elif section and ":" in line:
            key, value = line.split(":", 1)
            values = _yaml_list(value.strip())
            if section == "keypoint_names":
                keypoint_names[key.strip()] = values
            elif section == "class_alias":
                class_alias[key.strip()] = values
    return {
        "default_keypoints": default_keypoints,
        "keypoint_names": keypoint_names,
        "class_alias": class_alias,
    }


def _capabilities(classes: list[str], models: list[dict[str, Any]], pose_schema: dict[str, Any]) -> dict[str, Any]:
    class_set = {item.lower().replace("-", "_") for item in classes}
    pose_models = [item for item in models if item.get("task") == "pose"]
    detection_models = [item for item in models if item.get("task") == "detect"]
    cap_classes = sorted(item for item in class_set if "cap" in item or "lid" in item)
    container_classes = sorted(item for item in class_set if any(token in item for token in ("bottle", "tube", "beaker", "vial", "container", "flask")))
    tool_classes = sorted(item for item in class_set if item in {"pipette", "pipette_tip", "spatula", "spearhead"})
    panel_classes = sorted(item for item in class_set if any(token in item for token in ("panel", "button", "knob", "display", "screen", "balance")))
    liquid_classes = sorted(item for item in class_set if any(token in item for token in ("liquid", "fluid", "level", "meniscus", "stream")))
    return {
        "object_detection": {"available": bool(detection_models or models), "model_count": len(detection_models or models), "classes": classes},
        "pose_detection": {"available": bool(pose_models or pose_schema), "model_count": len(pose_models), "pose_schema": pose_schema},
        "hand_object_contact": {"available": bool({"gloved_hand", "hand"} & class_set and (set(container_classes) or set(tool_classes)))},
        "container_detection": {"available": bool(container_classes), "classes": container_classes},
        "cap_lid_detection": {"available": bool(cap_classes), "classes": cap_classes},
        "pipette_tool_detection": {"available": bool("pipette" in class_set or "pipette_tip" in class_set), "classes": tool_classes},
        "equipment_panel_detection": {"available": bool(panel_classes), "classes": panel_classes},
        "equipment_control_state_detection": {
            "available": bool({"button", "knob", "display", "screen", "panel"} & class_set),
            "classes": sorted({"button", "knob", "display", "screen", "panel"} & class_set),
        },
        "liquid_stream_segmentation": {"available": bool(liquid_classes), "classes": liquid_classes},
    }


def _canonical_classes(class_schema: dict[str, Any], datasets: list[dict[str, Any]], runtime_config: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for item in class_schema.get("classes") or []:
        if isinstance(item, dict) and item.get("name"):
            values.append(str(item["name"]))
    for dataset in datasets:
        values.extend(str(item) for item in dataset.get("class_names") or [])
    for key in ("detection", "class_registry"):
        if isinstance(runtime_config.get(key), list):
            values.extend(str(item) for item in runtime_config.get(key) or [])
    allowed = runtime_config.get("detection")
    if isinstance(allowed, dict):
        values.extend(str(item) for item in allowed.get("allowed_labels") or [])
    return sorted({_canonical_class(value) for value in values if str(value).strip()})


def _canonical_class(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    if text == "spearhead":
        return "pipette_tip"
    return text


def _resolve_candidate_path(value: str | Path, root: Path) -> Path | None:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_dataset_root(labsopguard_root: Path, yaml_path: Path, configured_root: Any) -> Path | None:
    if configured_root:
        root = Path(str(configured_root))
        if root.exists():
            return root.resolve()
        for candidate in (labsopguard_root / root, yaml_path.parent / root):
            if candidate.exists():
                return candidate.resolve()
    if (yaml_path.parent / "images").exists() or (yaml_path.parent / "labels").exists():
        return yaml_path.parent.resolve()
    return None


def _resolve_dataset_split(dataset_root: Path | None, yaml_path: Path, split_value: Any) -> Path | None:
    if not split_value:
        return None
    split = Path(str(split_value))
    if split.exists():
        return split.resolve()
    candidates = []
    if dataset_root:
        candidates.append(dataset_root / split)
    candidates.append(yaml_path.parent / split)
    return _first_existing(candidates) or (dataset_root / split if dataset_root else None)


def _matching_label_dir(image_dir: Path | None) -> Path | None:
    if not image_dir:
        return None
    parts = list(image_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        candidate = Path(*parts)
        if candidate.exists():
            return candidate
    sibling = image_dir.parent.parent / "labels" / image_dir.name
    return sibling if sibling.exists() else None


def _count_files(path: Path | None, extensions: set[str]) -> int:
    if not path or not path.exists():
        return 0
    try:
        return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)
    except OSError:
        return 0


def _yaml_value(value: str) -> Any:
    value = value.strip().strip("'\"")
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        return _yaml_list(value)
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _yaml_list(value: str) -> list[str]:
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    if not stripped:
        return []
    return [item.strip().strip("'\"") for item in stripped.split(",") if item.strip()]

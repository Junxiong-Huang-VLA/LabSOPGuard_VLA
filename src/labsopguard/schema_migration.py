"""Schema version migration for experiment output data."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

CURRENT_VERSIONS = {
    "physical_events": "physical_events.v4",
    "preprocessing": "preprocessing.v4",
    "presegment": "presegment.v1",
    "experiment_segmentation": "experiment_segmentation.v1",
    "published_materials": "published_materials.v1",
    "detection_cache": "detection_cache.v1",
    "global_material_index": "global_material_index.v1",
}

MigrationFunc = Callable[[Dict[str, Any]], Dict[str, Any]]
_MIGRATIONS: Dict[str, MigrationFunc] = {}


def register_migration(from_version: str, to_version: str):
    def decorator(func: MigrationFunc) -> MigrationFunc:
        _MIGRATIONS[from_version] = func
        func._target_version = to_version  # type: ignore
        return func
    return decorator


@register_migration("physical_events.v3", "physical_events.v4")
def _migrate_events_v3_to_v4(data: Dict[str, Any]) -> Dict[str, Any]:
    data["schema_version"] = "physical_events.v4"
    for event in data.get("events", []):
        event.setdefault("related_detection_classes", [])
        event.setdefault("notes", "")
    return data


@register_migration("preprocessing.v3", "preprocessing.v4")
def _migrate_preprocessing_v3_to_v4(data: Dict[str, Any]) -> Dict[str, Any]:
    data["schema_version"] = "preprocessing.v4"
    ep = data.get("event_preprocessing", {})
    ep.setdefault("tracked_object_count", 0)
    ep.setdefault("track_relation_count", 0)
    return data


def auto_migrate(data: Any) -> Any:
    """Apply all applicable migrations to bring data to current version."""
    if not data or not isinstance(data, dict):
        return data
    version = data.get("schema_version", "")
    for _ in range(10):
        if version not in _MIGRATIONS:
            break
        migration = _MIGRATIONS[version]
        target = getattr(migration, "_target_version", None)
        logger.info("Migrating schema: %s -> %s", version, target)
        data = migration(data)
        version = data.get("schema_version", "")
    return data


def migrate_file(file_path: str | Path) -> bool:
    """Read a JSON file, apply migrations if needed, write back."""
    path = Path(file_path)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    version_before = data.get("schema_version", "")
    migrated = auto_migrate(data)
    version_after = migrated.get("schema_version", "")
    if version_after != version_before:
        path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Migrated %s: %s -> %s", path.name, version_before, version_after)
        return True
    return False


def migrate_experiment_directory(experiment_dir: str | Path) -> Dict[str, bool]:
    """Migrate all JSON output files in an experiment directory."""
    exp_dir = Path(experiment_dir)
    results = {}
    for json_file in exp_dir.rglob("*.json"):
        if json_file.stat().st_size > 50_000_000:
            continue
        try:
            if migrate_file(json_file):
                results[str(json_file.relative_to(exp_dir))] = True
        except Exception as exc:
            logger.warning("Migration failed for %s: %s", json_file, exc)
    return results


def check_version_compatibility(data: Dict[str, Any], expected_prefix: str) -> bool:
    """Check if data's schema_version matches the expected current version."""
    version = data.get("schema_version", "")
    expected = CURRENT_VERSIONS.get(expected_prefix, "")
    return version == expected

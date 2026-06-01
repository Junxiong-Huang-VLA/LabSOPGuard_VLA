from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class WorkspaceEntry:
    path: str
    exists: bool
    role: str
    recommended_action: str
    reason: str
    size_bytes: int = 0


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def build_workspace_governance_report(workspace_root: str | Path) -> Dict[str, object]:
    root = Path(workspace_root)
    specs = [
        (
            root,
            "primary_project",
            "keep_as_primary_project",
            "Current source repository: backend, frontend, configs, scripts, src, tests, and pipeline code.",
        ),
        (
            root / "lab_preprocessing",
            "legacy_or_upstream_preprocessing_project",
            "migrate_unique_source_then_archive",
            "Legacy preprocessing code should be migrated into src/ or scripts/ before this folder is archived.",
        ),
        (
            root / ".ultralytics",
            "model_runtime_cache",
            "keep_ignored_runtime_cache",
            "Ultralytics runtime cache; keep ignored and do not treat as source or model storage.",
        ),
        (
            root / "memory",
            "experiment_memory_runtime_data",
            "keep_ignored_runtime_data",
            "Runtime memory data; keep ignored and controlled through runtime configuration.",
        ),
        (
            root / ".runtime",
            "runtime_outputs",
            "keep_ignored_runtime_outputs",
            "Stream buffers, daemon state, and temporary runtime artifacts; keep ignored.",
        ),
        (
            root / "outputs",
            "pipeline_outputs",
            "keep_ignored_or_externalize_heavy_outputs",
            "Generated experiment outputs; keep lightweight pointers in source and heavy artifacts outside git.",
        ),
    ]
    entries = [
        WorkspaceEntry(
            path=str(path),
            exists=path.exists(),
            role=role,
            recommended_action=action,
            reason=reason,
            size_bytes=_dir_size(path),
        )
        for path, role, action, reason in specs
    ]
    return {
        "schema_version": "workspace_governance.v1",
        "workspace_root": str(root),
        "entries": [asdict(entry) for entry in entries],
        "merge_policy": {
            "source_code": "Use src/, backend/, frontend/, scripts/, configs/, tests/, and docs/ for source work.",
            "external_assets": "Keep videos in LAB_VIDEO_STORE_ROOT, models in LAB_MODELS_DIR, and approved material libraries in LAB_MATERIAL_LIBRARY_ROOT.",
            "runtime_cache": "Keep .runtime/, .ultralytics/, memory/, and outputs/ ignored unless a small fixture is intentionally tracked.",
            "legacy_project": "Archive lab_preprocessing only after confirming unique code has moved into the current source tree.",
        },
    }


def write_workspace_governance_report(report: Dict[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

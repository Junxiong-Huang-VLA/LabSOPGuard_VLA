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
    entries: List[WorkspaceEntry] = []
    specs = [
        (
            root / "LabSOPGuard",
            "primary_project",
            "keep_as_primary_project",
            "当前主工程，包含后端、前端、配置、测试、运行入口和素材流水线。",
        ),
        (
            root / "lab_preprocessing",
            "legacy_or_upstream_preprocessing_project",
            "migrate_unique_source_then_archive",
            "如仍有独有源码，应迁入 LabSOPGuard/src 或 LabSOPGuard/scripts；重复产物应归档，不建议继续双项目并行开发。",
        ),
        (
            root / ".ultralytics",
            "model_runtime_cache",
            "keep_ignored_runtime_cache",
            "Ultralytics 运行缓存，不属于源码；应保留在项目根或通过 YOLO_CONFIG_DIR 指向 LabSOPGuard/.ultralytics，并保持 gitignore。",
        ),
        (
            root / "memory",
            "experiment_memory_runtime_data",
            "keep_ignored_runtime_data",
            "实验记忆/运行态数据，不属于源码；应通过运行配置管理并保持 gitignore。",
        ),
        (
            root / "LabSOPGuard" / ".runtime",
            "runtime_outputs",
            "keep_ignored_runtime_outputs",
            "压测、stream buffer、daemon 状态和临时产物目录，不进入版本库。",
        ),
    ]
    for path, role, action, reason in specs:
        entries.append(
            WorkspaceEntry(
                path=str(path),
                exists=path.exists(),
                role=role,
                recommended_action=action,
                reason=reason,
                size_bytes=_dir_size(path),
            )
        )
    return {
        "schema_version": "workspace_governance.v1",
        "workspace_root": str(root),
        "entries": [asdict(entry) for entry in entries],
        "merge_policy": {
            "source_code": "统一进入 LabSOPGuard/src、backend、frontend-app、scripts、configs、tests。",
            "runtime_cache": "保留在 .runtime、.ultralytics、memory，全部 gitignore。",
            "legacy_project": "lab_preprocessing 只在确认独有源码已迁移后归档，不自动删除。",
        },
    }


def write_workspace_governance_report(report: Dict[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

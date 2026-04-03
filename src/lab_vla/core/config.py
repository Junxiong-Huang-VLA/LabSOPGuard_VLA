from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def _norm_path(path_like: str | Path) -> str:
    return str(Path(path_like))


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class RuntimeConfig:
    root: Path
    raw: Dict[str, Any]

    @classmethod
    def from_file(cls, runtime_path: str | Path) -> "RuntimeConfig":
        p = Path(runtime_path)
        root = p.resolve().parents[2] if "configs" in p.parts else Path.cwd()
        raw = load_yaml(p)
        return cls(root=root, raw=raw)

    def resolve_path(self, rel_or_abs: str | Path) -> str:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return _norm_path(p)
        return _norm_path((self.root / p).resolve())

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


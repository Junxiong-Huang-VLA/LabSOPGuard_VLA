from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from project_name.common.io_utils import write_jsonl
from project_name.common.schemas import VLAResult


def export_results(path: str | Path, results: Iterable[VLAResult]) -> None:
    rows = [asdict(r) for r in results]
    write_jsonl(path, rows)

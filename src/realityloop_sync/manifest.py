from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import ConfigError, StreamConfig, SyncConfig


REQUIRED_MANIFEST_COLUMNS = {"camera_id", "stream_path"}


def resolve_streams(config: SyncConfig) -> tuple[SyncConfig, tuple[StreamConfig, ...], list[str]]:
    """Return effective config and camera streams from direct config or manifest CSV."""

    warnings: list[str] = []
    if not config.manifest_csv:
        if not config.streams:
            raise ConfigError("config must include streams or manifest_csv")
        return config, config.streams, warnings

    manifest_path = config.manifest_csv
    if not manifest_path.exists():
        raise ConfigError(f"manifest_csv does not exist: {manifest_path}")
    rows = load_manifest_rows(manifest_path)
    if config.run_id:
        scoped = [row for row in rows if not row.get("run_id") or str(row.get("run_id")) == config.run_id]
        if scoped:
            rows = scoped
    if not rows:
        raise ConfigError(f"manifest_csv has no usable rows for run_id={config.run_id!r}: {manifest_path}")

    first = rows[0]
    effective = replace(
        config,
        run_id=str(first.get("run_id") or config.run_id),
        experiment_name=str(first.get("experiment_name") or config.experiment_name),
        batch_id=str(first.get("batch_id") or config.batch_id),
    )
    streams = tuple(
        StreamConfig(
            camera_id=str(row["camera_id"]),
            stream_path=Path(str(row["stream_path"])),
            frames_csv=str(row.get("frames_csv") or "frames.csv"),
            filters={"stream_type": (str(row["stream_type"]),)} if str(row.get("stream_type") or "").strip() else {},
        )
        for row in rows
    )
    if config.streams:
        warnings.append("manifest_csv is present; direct streams were ignored")
    return effective, streams, warnings


def load_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    df = pd.read_csv(manifest_path, dtype=str).fillna("")
    missing = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ConfigError(f"manifest_csv is missing required columns: {', '.join(sorted(missing))}")
    rows = []
    for row in df.to_dict(orient="records"):
        camera_id = str(row.get("camera_id") or "").strip()
        stream_path = str(row.get("stream_path") or "").strip()
        if not camera_id or not stream_path:
            continue
        rows.append({key: str(value).strip() for key, value in row.items()})
    return rows


def build_manifest(
    *,
    video_database_root: str | Path,
    date: str,
    start_time: str,
    output: str | Path,
) -> Path:
    root = Path(video_database_root)
    rows = list(discover_camera_runs(root=root, date=date, start_time=start_time))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "experiment_name", "batch_id", "camera_id", "stream_path", "frames_csv", "stream_type"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def discover_camera_runs(*, root: Path, date: str, start_time: str) -> Iterable[dict[str, str]]:
    run_id = f"{date}_{start_time}"
    if not root.exists():
        raise ConfigError(f"video database root does not exist: {root}")
    for camera_root in sorted(path for path in root.iterdir() if path.is_dir()):
        camera_id = _camera_id_from_folder(camera_root.name)
        stream_path = camera_root / date / start_time
        if not stream_path.exists():
            continue
        yield {
            "run_id": run_id,
            "experiment_name": "",
            "batch_id": "",
            "camera_id": camera_id,
            "stream_path": str(stream_path),
            "frames_csv": "frames.csv",
            "stream_type": "",
        }


def _camera_id_from_folder(name: str) -> str:
    for token in reversed(name.replace("-", "_").split("_")):
        if token.lower().startswith("cam"):
            return token
    return name

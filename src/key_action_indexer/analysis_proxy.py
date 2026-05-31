from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from .schemas import SessionManifest, VideoSource

ANALYSIS_PROXY_SCHEMA_VERSION = "key_action_analysis_proxy.v1"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def analysis_proxy_enabled(*, default: bool = False) -> bool:
    for name in ("KEY_ACTION_FAST_LOCATE_ANALYSIS_PROXY", "KEY_ACTION_ANALYSIS_PROXY_ENABLED"):
        if os.environ.get(name) is not None:
            return _env_bool(name, default)
    return bool(default)


def analysis_proxy_settings() -> dict[str, Any]:
    width = max(160, _env_int("KEY_ACTION_ANALYSIS_PROXY_WIDTH", 640))
    fps = max(0.1, _env_float("KEY_ACTION_ANALYSIS_PROXY_FPS", 2.0))
    gop = max(1, _env_int("KEY_ACTION_ANALYSIS_PROXY_GOP", 1))
    return {
        "width": width,
        "fps": fps,
        "gop": gop,
        "codec": os.environ.get("KEY_ACTION_ANALYSIS_PROXY_CODEC", "libx264"),
        "preset": os.environ.get("KEY_ACTION_ANALYSIS_PROXY_PRESET", "veryfast"),
        "crf": max(1, min(51, _env_int("KEY_ACTION_ANALYSIS_PROXY_CRF", 32))),
        "pix_fmt": os.environ.get("KEY_ACTION_ANALYSIS_PROXY_PIX_FMT", "yuv420p"),
        "hwaccel": os.environ.get("KEY_ACTION_ANALYSIS_PROXY_HWACCEL", os.environ.get("KEY_ACTION_YOLO_FFMPEG_HWACCEL", "")),
        "hwaccel_output_format": os.environ.get(
            "KEY_ACTION_ANALYSIS_PROXY_HWACCEL_OUTPUT_FORMAT",
            os.environ.get("KEY_ACTION_YOLO_FFMPEG_HWACCEL_OUTPUT_FORMAT", ""),
        ),
        "audio": False,
    }


def _source_fingerprint(source: VideoSource) -> dict[str, Any]:
    path = Path(source.path)
    payload: dict[str, Any] = {
        "path": str(path.resolve(strict=False)),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        payload.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return payload


def _cache_key(source: VideoSource, settings: dict[str, Any]) -> str:
    payload = {
        "schema_version": ANALYSIS_PROXY_SCHEMA_VERSION,
        "source": _source_fingerprint(source),
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _proxy_source(source: VideoSource, proxy_path: Path, settings: dict[str, Any]) -> VideoSource:
    return replace(
        source,
        path=str(proxy_path),
        fps=float(settings["fps"]),
    )


def _ffmpeg_proxy_command(ffmpeg_path: str, source_path: Path, proxy_path: Path, settings: dict[str, Any]) -> list[str]:
    fps = float(settings["fps"])
    width = int(settings["width"])
    gop = int(settings["gop"])
    codec = str(settings["codec"] or "libx264")
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    hwaccel = str(settings.get("hwaccel") or "").strip().lower()
    if hwaccel and hwaccel not in {"0", "false", "no", "off", "none", "cpu"}:
        cmd.extend(["-hwaccel", hwaccel])
        output_format = str(settings.get("hwaccel_output_format") or "").strip()
        if output_format:
            cmd.extend(["-hwaccel_output_format", output_format])
    cmd.extend(
        [
            "-i",
            str(source_path),
            "-an",
            "-vf",
            f"scale={width}:-2,fps={fps:g}",
            "-c:v",
            codec,
            "-preset",
            str(settings["preset"] or "veryfast"),
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
        ]
    )
    if codec in {"h264_nvenc", "hevc_nvenc"}:
        cmd.extend(["-cq", str(int(settings["crf"]))])
    else:
        cmd.extend(["-crf", str(int(settings["crf"]))])
    pix_fmt = str(settings.get("pix_fmt") or "").strip()
    if pix_fmt:
        cmd.extend(["-pix_fmt", pix_fmt])
    cmd.append(str(proxy_path))
    return cmd


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def analysis_proxy_cache_paths(
    source: VideoSource,
    *,
    view: str,
    proxy_root: str | Path,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_settings = dict(settings or analysis_proxy_settings())
    cache_key = _cache_key(source, resolved_settings)
    view_root = Path(proxy_root) / view
    return {
        "settings": resolved_settings,
        "cache_key": cache_key,
        "view_root": view_root,
        "proxy_path": view_root / f"{cache_key[:16]}_analysis_proxy.mp4",
        "metadata_path": view_root / f"{cache_key[:16]}_analysis_proxy.json",
    }


def register_existing_analysis_proxy_for_source(
    source: VideoSource,
    *,
    view: str,
    proxy_root: str | Path,
    proxy_file: str | Path,
    source_label: str = "uploaded_analysis_proxy",
) -> dict[str, Any]:
    """Register a prebuilt proxy so coarse scan can use it without decoding raw video."""

    settings = analysis_proxy_settings()
    paths = analysis_proxy_cache_paths(source, view=view, proxy_root=proxy_root, settings=settings)
    proxy_path = Path(paths["proxy_path"])
    meta_path = Path(paths["metadata_path"])
    source_proxy = Path(proxy_file)
    if not source_proxy.exists() or source_proxy.stat().st_size <= 0:
        raise FileNotFoundError(f"analysis proxy file is missing or empty: {source_proxy}")
    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    if source_proxy.resolve(strict=False) != proxy_path.resolve(strict=False):
        temp_target = proxy_path.with_suffix(".uploading.mp4")
        try:
            if temp_target.exists():
                temp_target.unlink()
        except OSError:
            pass
        shutil.move(str(source_proxy), str(temp_target))
        temp_target.replace(proxy_path)
    meta = {
        "schema_version": ANALYSIS_PROXY_SCHEMA_VERSION,
        "enabled": True,
        "view": view,
        "source_path": str(Path(source.path)),
        "proxy_path": str(proxy_path),
        "metadata_path": str(meta_path),
        "cache_key": paths["cache_key"],
        "settings": settings,
        "source_fingerprint": _source_fingerprint(source),
        "status": "registered",
        "proxy_used": True,
        "source_label": source_label,
        "size_bytes": int(proxy_path.stat().st_size),
        "updated_at": time.time(),
    }
    _write_json(meta_path, meta)
    return meta


def build_analysis_proxy_for_source(
    source: VideoSource,
    *,
    view: str,
    proxy_root: str | Path,
    enabled: bool = True,
    dry_run: bool = False,
    existing_only: bool | None = None,
) -> tuple[VideoSource, dict[str, Any]]:
    settings = analysis_proxy_settings()
    root = Path(proxy_root)
    source_path = Path(source.path)
    cache_paths = analysis_proxy_cache_paths(source, view=view, proxy_root=root, settings=settings)
    cache_key = str(cache_paths["cache_key"])
    view_root = Path(cache_paths["view_root"])
    proxy_path = Path(cache_paths["proxy_path"])
    meta_path = Path(cache_paths["metadata_path"])
    base_meta: dict[str, Any] = {
        "schema_version": ANALYSIS_PROXY_SCHEMA_VERSION,
        "enabled": bool(enabled),
        "view": view,
        "source_path": str(source_path),
        "proxy_path": str(proxy_path),
        "metadata_path": str(meta_path),
        "cache_key": cache_key,
        "settings": settings,
        "source_fingerprint": _source_fingerprint(source),
    }
    timing_row = {
        "stage": "analysis_proxy_prepare",
        "pipeline_stage": "analysis_proxy_prepare",
        "source_view": view,
        "video_path": str(source_path),
        "sample_fps": float(settings["fps"]),
        "sampled_frames": 0,
        "read_frames": 0,
        "grab_frames": 0,
        "decode_sec": 0.0,
        "inference_sec": 0.0,
        "postprocess_sec": 0.0,
    }
    started = time.perf_counter()
    if not enabled:
        meta = {**base_meta, "status": "disabled", "proxy_used": False}
        return source, {**meta, "timing_row": {**timing_row, "wall_sec": 0.0, "scan_backend": "disabled"}}
    if dry_run:
        meta = {**base_meta, "status": "dry_run", "proxy_used": False}
        return source, {**meta, "timing_row": {**timing_row, "wall_sec": 0.0, "scan_backend": "dry_run"}}
    if not source_path.exists():
        wall = time.perf_counter() - started
        meta = {**base_meta, "status": "skipped_missing_source", "proxy_used": False, "wall_sec": round(wall, 6)}
        return source, {**meta, "timing_row": {**timing_row, "wall_sec": round(wall, 6), "scan_backend": "missing_source"}}
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        wall = time.perf_counter() - started
        meta = {**base_meta, "status": "skipped_no_ffmpeg", "proxy_used": False, "wall_sec": round(wall, 6)}
        return source, {**meta, "timing_row": {**timing_row, "wall_sec": round(wall, 6), "scan_backend": "no_ffmpeg"}}

    cached_meta: dict[str, Any] = {}
    if meta_path.exists() and proxy_path.exists() and proxy_path.stat().st_size > 0:
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            cached_meta = data if isinstance(data, dict) else {}
        except Exception:
            cached_meta = {}
        if cached_meta.get("cache_key") == cache_key and cached_meta.get("settings") == settings:
            wall = time.perf_counter() - started
            meta = {
                **base_meta,
                "status": "cache_hit",
                "proxy_used": True,
                "wall_sec": round(wall, 6),
                "size_bytes": int(proxy_path.stat().st_size),
            }
            return _proxy_source(source, proxy_path, settings), {
                **meta,
                "timing_row": {
                    **timing_row,
                    "wall_sec": round(wall, 6),
                    "scan_backend": "analysis_proxy_cache_hit",
                },
            }

    should_require_existing = (
        bool(existing_only)
        if existing_only is not None
        else _env_bool("KEY_ACTION_ANALYSIS_PROXY_EXISTING_ONLY", False)
    )
    if should_require_existing:
        wall = time.perf_counter() - started
        meta = {
            **base_meta,
            "status": "skipped_existing_proxy_missing",
            "proxy_used": False,
            "wall_sec": round(wall, 6),
            "reason": "existing_only_cache_miss",
        }
        return source, {
            **meta,
            "timing_row": {
                **timing_row,
                "wall_sec": round(wall, 6),
                "scan_backend": "analysis_proxy_existing_only_cache_miss",
            },
        }

    view_root.mkdir(parents=True, exist_ok=True)
    temp_path = proxy_path.with_suffix(".tmp.mp4")
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError:
        pass
    cmd = _ffmpeg_proxy_command(ffmpeg_path, source_path, temp_path, settings)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        temp_path.replace(proxy_path)
        wall = time.perf_counter() - started
        meta = {
            **base_meta,
            "status": "built",
            "proxy_used": True,
            "wall_sec": round(wall, 6),
            "size_bytes": int(proxy_path.stat().st_size) if proxy_path.exists() else 0,
            "ffmpeg_command": cmd,
            "updated_at": time.time(),
        }
        _write_json(meta_path, meta)
        return _proxy_source(source, proxy_path, settings), {
            **meta,
            "timing_row": {
                **timing_row,
                "wall_sec": round(wall, 6),
                "decode_sec": round(wall, 6),
                "scan_backend": "analysis_proxy_ffmpeg",
            },
        }
    except Exception as exc:
        wall = time.perf_counter() - started
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        meta = {
            **base_meta,
            "status": "failed",
            "proxy_used": False,
            "wall_sec": round(wall, 6),
            "error": str(exc),
        }
        return source, {
            **meta,
            "timing_row": {
                **timing_row,
                "wall_sec": round(wall, 6),
                "scan_backend": "analysis_proxy_failed",
                "errors": str(exc),
            },
        }


def build_analysis_proxies(
    manifest: SessionManifest,
    *,
    proxy_root: str | Path,
    views: list[str],
    enabled: bool = True,
    dry_run: bool = False,
    existing_only: bool | None = None,
) -> tuple[dict[str, VideoSource], dict[str, Any]]:
    sources = manifest.videos.all_sources()
    requested = [view for view in views if view in sources]
    worker_count = max(1, min(len(requested) or 1, _env_int("KEY_ACTION_ANALYSIS_PROXY_WORKERS", len(requested) or 1)))
    started = time.perf_counter()
    proxy_sources: dict[str, VideoSource] = {}
    view_meta: dict[str, dict[str, Any]] = {}

    def build(view: str) -> tuple[str, VideoSource, dict[str, Any]]:
        proxy_source, meta = build_analysis_proxy_for_source(
            sources[view],
            view=view,
            proxy_root=proxy_root,
            enabled=enabled,
            dry_run=dry_run,
            existing_only=existing_only,
        )
        return view, proxy_source, meta

    if worker_count <= 1:
        for view in requested:
            view_id, proxy_source, meta = build(view)
            proxy_sources[view_id] = proxy_source
            view_meta[view_id] = meta
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="analysis_proxy") as executor:
            futures = [executor.submit(build, view) for view in requested]
            for future in as_completed(futures):
                view_id, proxy_source, meta = future.result()
                proxy_sources[view_id] = proxy_source
                view_meta[view_id] = meta

    for view, source in sources.items():
        proxy_sources.setdefault(view, source)

    elapsed = time.perf_counter() - started
    timing_rows = []
    for meta in view_meta.values():
        row = meta.get("timing_row") if isinstance(meta.get("timing_row"), dict) else None
        if row:
            row = dict(row)
            row["stage_parallel_elapsed_sec"] = round(elapsed, 6)
            row["stage_parallel_workers"] = int(worker_count)
            row["stage_scan_task_count"] = int(len(requested))
            timing_rows.append(row)
    summary = {
        "schema_version": ANALYSIS_PROXY_SCHEMA_VERSION,
        "enabled": bool(enabled),
        "proxy_root": str(Path(proxy_root)),
        "views": view_meta,
        "settings": analysis_proxy_settings(),
        "parallel_workers": int(worker_count),
        "parallel_enabled": bool(worker_count > 1),
        "requested_views": requested,
        "elapsed_sec": round(elapsed, 6),
        "proxy_used": any(bool(meta.get("proxy_used")) for meta in view_meta.values()),
        "timing_rows": timing_rows,
    }
    return proxy_sources, summary


def analysis_proxy_cache_payload(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict) or not summary:
        return {"schema_version": ANALYSIS_PROXY_SCHEMA_VERSION, "enabled": False}
    views = {}
    for view, meta in (summary.get("views") or {}).items():
        if not isinstance(meta, dict):
            continue
        views[str(view)] = {
            "source_path": meta.get("source_path"),
            "cache_key": meta.get("cache_key"),
            "settings": meta.get("settings"),
            "proxy_path": meta.get("proxy_path") if meta.get("proxy_used") else None,
            "proxy_used": bool(meta.get("proxy_used")),
            "status_class": "proxy" if meta.get("proxy_used") else str(meta.get("status") or "unknown"),
        }
    return {
        "schema_version": ANALYSIS_PROXY_SCHEMA_VERSION,
        "enabled": bool(summary.get("enabled")),
        "settings": summary.get("settings"),
        "views": views,
    }

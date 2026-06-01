"""Measure user-facing upload-to-results latency for the key-action pipeline.

The benchmark intentionally drives the same HTTP path as the frontend:
create experiment -> upload dual-view videos -> wait until the
workspace-visible artifacts are available.  The primary number is end-to-end
wall time from the client's perspective; internal algorithm timings are
recorded only as breakdown fields.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from requests_toolbelt import MultipartEncoder


DEFAULT_BASE_URL = "http://127.0.0.1:5173/api/v1"
FINAL_STATUSES = {"completed", "partial_failed", "failed"}
RUN_HISTORY_LIMIT = 50
METRIC_KEYS = [
    "client_end_to_end_sec",
    "client_create_sec",
    "client_upload_http_sec",
    "client_raw_upload_sec",
    "client_run_request_sec",
    "client_queue_start_sec",
    "client_analysis_poll_wait_sec",
    "client_result_fetch_sec",
    "client_sub_experiments_fetch_sec",
    "client_materials_fetch_sec",
    "client_analysis_wait_sec",
    "server_end_to_end_sec",
    "upload_save_sec",
    "queue_wait_sec",
    "algorithm_elapsed_sec",
    "core_analysis_sec",
]
METRIC_LABELS_ZH = {
    "client_end_to_end_sec": "用户上传到结果可见",
    "client_create_sec": "创建实验",
    "client_upload_http_sec": "上传/落盘",
    "client_raw_upload_sec": "双视角并行上传",
    "client_run_request_sec": "启动分析请求",
    "client_queue_start_sec": "排队/启动",
    "client_analysis_poll_wait_sec": "等待服务端完成",
    "client_result_fetch_sec": "结果拉取",
    "client_sub_experiments_fetch_sec": "实验片段拉取",
    "client_materials_fetch_sec": "素材发布",
    "client_analysis_wait_sec": "等待分析结果",
    "server_end_to_end_sec": "服务端端到端",
    "upload_save_sec": "服务端接收/落盘",
    "queue_wait_sec": "服务端排队等待",
    "algorithm_elapsed_sec": "算法分析",
    "core_analysis_sec": "核心分析",
}


class StreamingMultipartBody:
    """A small streaming multipart encoder for very large local video uploads."""

    def __init__(self, fields: dict[str, str], files: dict[str, tuple[str, Path, str]]) -> None:
        self.boundary = f"----labembodied-{uuid.uuid4().hex}"
        self.content_type = f"multipart/form-data; boundary={self.boundary}"
        self._parts: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        for name, value in fields.items():
            header = (
                f"--{self.boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            ).encode("utf-8")
            payload = str(value).encode("utf-8")
            self._parts.append({"header": header, "payload": payload, "footer": b"\r\n", "size": len(header) + len(payload) + 2})
        for name, (filename, path, content_type) in files.items():
            header = (
                f"--{self.boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
            handle = path.open("rb")
            self._handles.append(handle)
            file_size = path.stat().st_size
            self._parts.append({"header": header, "payload": handle, "footer": b"\r\n", "size": len(header) + file_size + 2})
        self._closing = f"--{self.boundary}--\r\n".encode("utf-8")
        self.content_length = sum(int(part["size"]) for part in self._parts) + len(self._closing)
        self._part_index = 0
        self._section: str = "header"
        self._buffer = b""
        self._closed = False

    def __len__(self) -> int:
        return self.content_length

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handle in self._handles:
            try:
                handle.close()
            except Exception:
                pass

    def _read_next(self, size: int) -> bytes:
        if self._part_index >= len(self._parts):
            if self._section != "closing":
                self._section = "closing"
                self._buffer = self._closing
            if not self._buffer:
                return b""
            chunk = self._buffer[:size]
            self._buffer = self._buffer[size:]
            return chunk

        part = self._parts[self._part_index]
        if self._section in {"header", "payload_bytes", "footer"}:
            if not self._buffer:
                if self._section == "header":
                    self._buffer = part["header"]
                    self._section = "payload"
                elif self._section == "payload_bytes":
                    self._buffer = part["payload"]
                    self._section = "footer"
                else:
                    self._buffer = part["footer"]
                    self._section = "header"
                    self._part_index += 1
            chunk = self._buffer[:size]
            self._buffer = self._buffer[size:]
            return chunk

        payload = part["payload"]
        if hasattr(payload, "read"):
            chunk = payload.read(size)
            if chunk:
                return chunk
            self._section = "footer"
            return self._read_next(size)
        self._section = "payload_bytes"
        return self._read_next(size)

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            return b""
        if size is None or size < 0:
            chunks: list[bytes] = []
            while True:
                chunk = self._read_next(1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        if size == 0:
            return b""
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = self._read_next(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * max(0.0, min(1.0, p / 100.0))
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return round(ordered[lower], 3)
    weight = rank - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 3)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def run_metric_value(row: dict[str, Any], key: str) -> float | None:
    if key == "client_queue_start_sec":
        run_request = run_metric_value(row, "client_run_request_sec")
        queue_wait = run_metric_value(row, "queue_wait_sec")
        if run_request is None and queue_wait is None:
            return None
        return round((run_request or 0.0) + (queue_wait or 0.0), 3)
    own = finite_float(row.get(key))
    if own is not None:
        return own
    timing = row.get("backend_timing") if isinstance(row.get("backend_timing"), dict) else {}
    return finite_float(timing.get(key))


def summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [value for row in rows if (value := run_metric_value(row, key)) is not None]
    latest = values[-1] if values else None
    return {
        "sample_count": len(values),
        "p50_sec": percentile(values, 50),
        "p90_sec": percentile(values, 90),
        "mean_sec": round(statistics.mean(values), 3) if values else None,
        "best_sec": round(min(values), 3) if values else None,
        "latest_sec": round(latest, 3) if latest is not None else None,
        "max_sec": round(max(values), 3) if values else None,
    }


def build_target_status(metrics: dict[str, dict[str, Any]], *, target_p50_sec: float, target_p90_sec: float) -> dict[str, Any]:
    user_metric = metrics.get("client_end_to_end_sec") or {}
    p50 = finite_float(user_metric.get("p50_sec"))
    p90 = finite_float(user_metric.get("p90_sec"))
    p50_ok = p50 is not None and p50 <= target_p50_sec
    p90_ok = p90 is not None and p90 <= target_p90_sec
    return {
        "metric": "client_end_to_end_sec",
        "target_p50_sec": float(target_p50_sec),
        "target_p90_sec": float(target_p90_sec),
        "actual_p50_sec": p50,
        "actual_p90_sec": p90,
        "p50_ok": p50_ok,
        "p90_ok": p90_ok,
        "passed": bool(p50_ok and p90_ok),
    }


def build_pipeline_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready_rows = [row for row in rows if row.get("ready_for_demo")]
    checked_rows = ready_rows or rows
    segment_counts = [int(row.get("segment_count") or 0) for row in checked_rows]
    dual_view_counts = [int(row.get("dual_view_segment_count") or 0) for row in checked_rows]
    material_counts = [int(row.get("published_material_count") or 0) for row in checked_rows]
    has_group_metrics = any("published_material_group_count" in row for row in checked_rows)
    material_group_counts = [int(row.get("published_material_group_count") or 0) for row in checked_rows if "published_material_group_count" in row]
    complete_group_counts = [
        int(row.get("complete_dual_view_material_group_count") or 0)
        for row in checked_rows
        if "complete_dual_view_material_group_count" in row
    ]
    quality_gates: dict[str, Any] = {
        "ready_for_demo_count": len(ready_rows),
        "run_count": len(rows),
        "all_runs_ready_for_demo": bool(rows) and len(ready_rows) == len(rows),
        "min_segment_count": min(segment_counts) if segment_counts else 0,
        "min_dual_view_segment_count": min(dual_view_counts) if dual_view_counts else 0,
        "min_published_material_count": min(material_counts) if material_counts else 0,
        "all_segments_have_dual_view": bool(checked_rows)
        and all(
            int(row.get("segment_count") or 0) > 0
            and int(row.get("dual_view_segment_count") or 0) == int(row.get("segment_count") or 0)
            for row in checked_rows
        ),
    }
    if has_group_metrics:
        quality_gates.update(
            {
                "min_published_material_group_count": min(material_group_counts) if material_group_counts else 0,
                "min_complete_dual_view_material_group_count": min(complete_group_counts) if complete_group_counts else 0,
                "all_material_groups_have_dual_view_keyframe_and_clip": bool(checked_rows)
                and all(
                    int(row.get("published_material_group_count") or 0) > 0
                    and int(row.get("complete_dual_view_material_group_count") or 0)
                    == int(row.get("published_material_group_count") or 0)
                    for row in checked_rows
                ),
            }
        )
    return {
        "schema_version": "dual_view_key_action_pipeline_contract.v1",
        "steps": [
            {"step": "dual_view_upload", "label_zh": "双视角上传/落盘", "metric": "client_upload_http_sec"},
            {"step": "timestamp_alignment", "label_zh": "双视角时间戳对齐", "condition": "dual_view_segment_count == segment_count"},
            {"step": "experiment_segment_selection", "label_zh": "连续视频流筛选 N 段实验", "metric": "segment_count"},
            {"step": "physical_action_scan", "label_zh": "实验片段精扫物理动作交互", "metric": "core_analysis_sec"},
            {"step": "material_generation", "label_zh": "关键帧/关键片段生成", "condition": "published_material_count > 0"},
            {"step": "material_library_sync", "label_zh": "关键素材库与独立文件夹同步", "condition": "ready_for_demo == true"},
        ],
        "quality_gates": quality_gates,
    }


def run_history_key(row: dict[str, Any]) -> str:
    experiment_id = str(row.get("experiment_id") or "").strip()
    if experiment_id:
        return f"experiment:{experiment_id}"
    title = str(row.get("title") or "").strip()
    created_at = str(row.get("created_at") or "").strip()
    return f"run:{created_at}:{title}"


def dedupe_run_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    index_by_key: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = run_history_key(row)
        if key in index_by_key:
            ordered[index_by_key[key]] = row
            continue
        index_by_key[key] = len(ordered)
        ordered.append(row)
    return ordered[-RUN_HISTORY_LIMIT:]


def load_previous_run_history(latest_path: Path) -> list[dict[str, Any]]:
    if not latest_path.exists():
        return []
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("run_history", "runs"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    latest_run = payload.get("latest_run")
    if isinstance(latest_run, dict):
        rows.append(latest_run)
    return dedupe_run_history(rows)


def build_summary(
    *,
    base_url: str,
    first_path: Path,
    third_path: Path,
    results: list[dict[str, Any]],
    history_results: list[dict[str, Any]] | None = None,
    target_p50_sec: float,
    target_p90_sec: float,
) -> dict[str, Any]:
    run_history = dedupe_run_history([*(history_results or []), *results])
    ready_rows = [row for row in run_history if row.get("ready_for_demo")]
    current_ready_rows = [row for row in results if row.get("ready_for_demo")]
    metrics = {key: summarize_metric(ready_rows, key) for key in METRIC_KEYS}
    return {
        "schema_version": "upload_e2e_benchmark.v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": str(base_url).rstrip("/"),
        "first_person_video": str(first_path),
        "third_person_video": str(third_path),
        "upload_modes": sorted({str(row.get("upload_mode") or "unknown") for row in run_history}),
        "repeat_count": len(results),
        "current_repeat_count": len(results),
        "successful_ready_count": len(ready_rows),
        "current_successful_ready_count": len(current_ready_rows),
        "history_count": len(run_history),
        "history_ready_count": len(ready_rows),
        "metric_scope": "real_upload_e2e_run_history",
        "metric_scope_zh": "真实上传端到端历史；从创建实验、上传、落盘、排队、分析、结果拉取到素材发布，不取算法最好值。",
        "client_end_to_end_sec": {
            "p50": metrics["client_end_to_end_sec"]["p50_sec"],
            "p90": metrics["client_end_to_end_sec"]["p90_sec"],
            "mean": metrics["client_end_to_end_sec"]["mean_sec"],
            "min": metrics["client_end_to_end_sec"]["best_sec"],
            "max": metrics["client_end_to_end_sec"]["max_sec"],
        },
        "metric_labels_zh": METRIC_LABELS_ZH,
        "metrics": metrics,
        "target_status": build_target_status(metrics, target_p50_sec=target_p50_sec, target_p90_sec=target_p90_sec),
        "pipeline_contract": build_pipeline_contract(run_history),
        "latest_run": ready_rows[-1] if ready_rows else (run_history[-1] if run_history else None),
        "current_runs": results,
        "run_history": run_history,
        "runs": run_history,
    }


def compact_run_timing(overview: dict[str, Any]) -> dict[str, Any]:
    run = overview.get("run") if isinstance(overview.get("run"), dict) else {}
    timing = run.get("timing") if isinstance(run.get("timing"), dict) else {}
    return {
        "user_visible_status": run.get("status"),
        "algorithm_elapsed_sec": timing.get("algorithm_elapsed_sec") or timing.get("elapsed_sec") or run.get("elapsed_sec"),
        "server_end_to_end_sec": timing.get("server_end_to_end_sec"),
        "upload_save_sec": timing.get("upload_save_sec"),
        "queue_wait_sec": timing.get("queue_wait_sec"),
        "core_analysis_sec": timing.get("core_analysis_sec") or run.get("core_analysis_sec"),
        "stage_rows": timing.get("stages") or [],
    }


def upload_save_sec_from_upload_result(upload_result: dict[str, Any]) -> float | None:
    candidates: list[float] = []
    for key in ("first_upload", "third_upload", "run_response"):
        payload = upload_result.get(key)
        if not isinstance(payload, dict):
            continue
        value = finite_float(payload.get("upload_save_sec"))
        if value is not None:
            candidates.append(value)
    return round(max(candidates), 3) if candidates else None


def count_complete_dual_view_material_groups(published: dict[str, Any]) -> tuple[int, int]:
    groups = published.get("grouped_items") if isinstance(published.get("grouped_items"), list) else []
    group_count = 0
    complete_count = 0
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_count += 1
        views = set(str(view) for view in (group.get("views") or []) if str(view).strip())
        has_dual_view = group.get("view") == "dual_view" or {"first_person", "third_person"}.issubset(views)
        clip_count = int(group.get("clip_count") or 0)
        keyframe_count = int(group.get("keyframe_count") or 0)
        if has_dual_view and clip_count >= 2 and keyframe_count >= 2:
            complete_count += 1
    return group_count, complete_count


def api_get(
    session: requests.Session,
    base_url: str,
    path: str,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    retry_sleep_sec: float = 2.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            response = session.get(f"{base_url}{path}", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt >= max(1, retries):
                break
            time.sleep(max(0.1, retry_sleep_sec))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"GET {path} failed without response")


def upload_raw_view(
    *,
    base_url: str,
    experiment_id: str,
    view: str,
    path: Path,
    headers: dict[str, str],
    timeout_sec: float,
) -> dict[str, Any]:
    upload_headers = {
        **headers,
        "Content-Type": "video/mp4",
        "Content-Length": str(path.stat().st_size),
        "X-Filename": quote(path.name),
    }
    with path.open("rb") as handle:
        response = requests.put(
            f"{base_url}/experiments/{experiment_id}/key-actions/upload-raw/{view}",
            data=handle,
            headers=upload_headers,
            timeout=max(60.0, timeout_sec),
        )
    response.raise_for_status()
    return response.json()


def upload_and_start_raw(
    *,
    session: requests.Session,
    base_url: str,
    experiment_id: str,
    first_path: Path,
    third_path: Path,
    session_start_time: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    upload_started = time.perf_counter()
    inherited_headers = {str(key): str(value) for key, value in session.headers.items()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            upload_raw_view,
            base_url=base_url,
            experiment_id=experiment_id,
            view="first_person",
            path=first_path,
            headers=inherited_headers,
            timeout_sec=timeout_sec,
        )
        third_future = pool.submit(
            upload_raw_view,
            base_url=base_url,
            experiment_id=experiment_id,
            view="third_person",
            path=third_path,
            headers=inherited_headers,
            timeout_sec=timeout_sec,
        )
        first_result = first_future.result()
        third_result = third_future.result()
    raw_upload_sec = time.perf_counter() - upload_started

    run_started = time.perf_counter()
    run_response = session.post(
        f"{base_url}/experiments/{experiment_id}/key-actions/run",
        json={
            "session_start_time": session_start_time,
            "force": True,
        },
        timeout=60,
    )
    run_response.raise_for_status()
    run_request_sec = time.perf_counter() - run_started
    return {
        "upload_mode": "raw_stream",
        "client_raw_upload_sec": raw_upload_sec,
        "client_run_request_sec": run_request_sec,
        "client_upload_http_sec": raw_upload_sec + run_request_sec,
        "first_upload": first_result,
        "third_upload": third_result,
        "run_response": run_response.json(),
    }


def upload_and_start_multipart(
    *,
    session: requests.Session,
    base_url: str,
    experiment_id: str,
    first_path: Path,
    third_path: Path,
    session_start_time: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    fields: dict[str, str] = {}
    if session_start_time:
        fields["session_start_time"] = session_start_time
    first_handle = first_path.open("rb")
    third_handle = third_path.open("rb")
    multipart_fields: dict[str, Any] = {
        **fields,
        "first_person_video": (first_path.name, first_handle, "video/mp4"),
        "third_person_video": (third_path.name, third_handle, "video/mp4"),
    }
    body = MultipartEncoder(fields=multipart_fields)
    started = time.perf_counter()
    try:
        upload_response = session.post(
            f"{base_url}/experiments/{experiment_id}/key-actions/upload-and-run",
            data=body,
            headers={
                "Content-Type": body.content_type,
            },
            timeout=max(60.0, timeout_sec),
        )
    finally:
        first_handle.close()
        third_handle.close()
    upload_response.raise_for_status()
    return {
        "upload_mode": "multipart",
        "client_upload_http_sec": time.perf_counter() - started,
        "run_response": upload_response.json(),
    }


def run_once(
    *,
    session: requests.Session,
    base_url: str,
    first_path: Path,
    third_path: Path,
    title_prefix: str,
    session_start_time: str | None,
    poll_interval_sec: float,
    timeout_sec: float,
    run_index: int,
    upload_mode: str,
) -> dict[str, Any]:
    client_started_at = time.perf_counter()
    created_at_iso = datetime.now(timezone.utc).isoformat()
    title = f"{title_prefix} {datetime.now().strftime('%Y%m%d-%H%M%S')}-{run_index:02d}"

    create_start = time.perf_counter()
    create_response = session.post(
        f"{base_url}/experiments",
        json={
            "title": title,
            "description": "E2E upload benchmark: user-facing upload-to-results latency",
            "context_text": "benchmark_upload_e2e",
        },
        timeout=30,
    )
    create_response.raise_for_status()
    experiment_id = str(create_response.json()["experiment_id"])
    create_sec = time.perf_counter() - create_start

    if upload_mode == "raw":
        upload_result = upload_and_start_raw(
            session=session,
            base_url=base_url,
            experiment_id=experiment_id,
            first_path=first_path,
            third_path=third_path,
            session_start_time=session_start_time,
            timeout_sec=timeout_sec,
        )
    elif upload_mode == "multipart":
        upload_result = upload_and_start_multipart(
            session=session,
            base_url=base_url,
            experiment_id=experiment_id,
            first_path=first_path,
            third_path=third_path,
            session_start_time=session_start_time,
            timeout_sec=timeout_sec,
        )
    else:
        raise ValueError(f"Unsupported upload mode: {upload_mode}")
    upload_sec = float(upload_result.get("client_upload_http_sec") or 0.0)

    analysis_wait_start = time.perf_counter()
    deadline = time.perf_counter() + timeout_sec
    overview: dict[str, Any] = {}
    status = "unknown"
    while time.perf_counter() < deadline:
        try:
            overview = api_get(
                session,
                base_url,
                f"/experiments/{experiment_id}/analysis-overview",
                timeout=30,
                retries=2,
                retry_sleep_sec=poll_interval_sec,
            )
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(poll_interval_sec)
            continue
        run = overview.get("run") if isinstance(overview.get("run"), dict) else {}
        status = str(run.get("status") or "unknown")
        if status in FINAL_STATUSES:
            break
        time.sleep(poll_interval_sec)
    else:
        status = "timeout"

    poll_wait_sec = time.perf_counter() - analysis_wait_start
    sub_fetch_start = time.perf_counter()
    sub_experiments = api_get(
        session,
        base_url,
        f"/experiments/{experiment_id}/sub-experiments",
        timeout=120,
        retries=5,
        retry_sleep_sec=3.0,
    )
    sub_fetch_sec = time.perf_counter() - sub_fetch_start
    materials_fetch_start = time.perf_counter()
    published = api_get(
        session,
        base_url,
        f"/experiments/{experiment_id}/materials/published",
        timeout=120,
        retries=5,
        retry_sleep_sec=3.0,
    )
    materials_fetch_sec = time.perf_counter() - materials_fetch_start
    result_fetch_sec = sub_fetch_sec + materials_fetch_sec
    published_material_group_count, complete_dual_view_material_group_count = count_complete_dual_view_material_groups(published)
    client_end_to_end_sec = time.perf_counter() - client_started_at

    timing = compact_run_timing(overview)
    upload_save_sec = upload_save_sec_from_upload_result(upload_result)
    queue_wait_sec = finite_float(timing.get("queue_wait_sec")) or 0.0
    run_request_sec = finite_float(upload_result.get("client_run_request_sec"))
    if upload_save_sec is not None and timing.get("upload_save_sec") is None:
        timing["upload_save_sec"] = upload_save_sec
        stage_rows = timing.get("stage_rows")
        if isinstance(stage_rows, list) and not any(
            isinstance(row, dict) and row.get("stage") == "server_upload_save" for row in stage_rows
        ):
            timing["stage_rows"] = [
                {
                    "stage": "server_upload_save",
                    "stage_label_zh": "服务端接收/落盘",
                    "duration_sec": upload_save_sec,
                },
                *stage_rows,
            ]
    segment_count = int(sub_experiments.get("total") or 0)
    segment_video_count = sum(
        1
        for segment in sub_experiments.get("segments") or []
        if isinstance(segment, dict) and (segment.get("first_person_video_url") or segment.get("third_person_video_url"))
    )
    dual_view_segment_count = sum(
        1
        for segment in sub_experiments.get("segments") or []
        if isinstance(segment, dict) and segment.get("first_person_video_url") and segment.get("third_person_video_url")
    )
    return {
        "experiment_id": experiment_id,
        "title": title,
        "created_at": created_at_iso,
        "status": status,
        "client_end_to_end_sec": round(client_end_to_end_sec, 3),
        "client_create_sec": round(create_sec, 3),
        "client_upload_http_sec": round(upload_sec, 3),
        "client_raw_upload_sec": round(float(upload_result.get("client_raw_upload_sec") or 0.0), 3)
        if upload_result.get("client_raw_upload_sec") is not None
        else None,
        "client_run_request_sec": round(float(upload_result.get("client_run_request_sec") or 0.0), 3)
        if upload_result.get("client_run_request_sec") is not None
        else None,
        "client_queue_start_sec": round((run_request_sec or 0.0) + queue_wait_sec, 3)
        if run_request_sec is not None or queue_wait_sec
        else None,
        "client_analysis_poll_wait_sec": round(poll_wait_sec, 3),
        "client_result_fetch_sec": round(result_fetch_sec, 3),
        "client_sub_experiments_fetch_sec": round(sub_fetch_sec, 3),
        "client_materials_fetch_sec": round(materials_fetch_sec, 3),
        "client_analysis_wait_sec": round(time.perf_counter() - analysis_wait_start, 3),
        "upload_save_sec": upload_save_sec,
        "upload_mode": str(upload_result.get("upload_mode") or upload_mode),
        "segment_count": segment_count,
        "segment_video_count": segment_video_count,
        "dual_view_segment_count": dual_view_segment_count,
        "published_material_count": int(published.get("total") or len(published.get("items") or [])),
        "published_material_group_count": published_material_group_count,
        "complete_dual_view_material_group_count": complete_dual_view_material_group_count,
        "hidden_incomplete_material_count": int(
            (published.get("dual_view_quality_gate") or {}).get("hidden_item_count") or 0
        )
        if isinstance(published.get("dual_view_quality_gate"), dict)
        else 0,
        "backend_timing": timing,
        "ready_for_demo": (
            status in {"completed", "partial_failed"}
            and segment_count > 0
            and dual_view_segment_count == segment_count
            and int(published.get("total") or len(published.get("items") or [])) > 0
            and published_material_group_count > 0
            and complete_dual_view_material_group_count == published_material_group_count
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark upload-to-results E2E latency.")
    parser.add_argument("--first", required=True, type=Path, help="First-person RGB video path")
    parser.add_argument("--third", required=True, type=Path, help="Third-person RGB video path")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    parser.add_argument("--session-start-time", default=None)
    parser.add_argument("--title-prefix", default="E2E latency benchmark")
    parser.add_argument("--upload-mode", choices=["raw", "multipart"], default="raw")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--target-p50-sec", type=float, default=45.0)
    parser.add_argument("--target-p90-sec", type=float, default=60.0)
    args = parser.parse_args()

    first_path = args.first.resolve()
    third_path = args.third.resolve()
    if not first_path.exists():
        raise SystemExit(f"first-person video not found: {first_path}")
    if not third_path.exists():
        raise SystemExit(f"third-person video not found: {third_path}")

    session = requests.Session()
    session.headers.update({"X-Operator-Role": "admin"})
    results: list[dict[str, Any]] = []
    out_path = args.out
    write_partials = out_path is not None
    if out_path is None:
        out_path = Path(__file__).resolve().parents[1] / "outputs" / "benchmarks" / f"upload_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    latest_path = out_path.parent / "upload_e2e_latest.json"
    previous_history = load_previous_run_history(latest_path)
    for index in range(1, max(1, args.repeats) + 1):
        result = run_once(
            session=session,
            base_url=str(args.base_url).rstrip("/"),
            first_path=first_path,
            third_path=third_path,
            title_prefix=args.title_prefix,
            session_start_time=args.session_start_time,
            poll_interval_sec=args.poll_interval_sec,
            timeout_sec=args.timeout_sec,
            run_index=index,
            upload_mode=args.upload_mode,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if write_partials:
            partial_path = out_path.with_suffix(f".partial.{index:02d}.json")
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(
                json.dumps(
                    build_summary(
                        base_url=str(args.base_url).rstrip("/"),
                        first_path=first_path,
                        third_path=third_path,
                        results=results,
                        history_results=previous_history,
                        target_p50_sec=args.target_p50_sec,
                        target_p90_sec=args.target_p90_sec,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    summary = build_summary(
        base_url=str(args.base_url).rstrip("/"),
        first_path=first_path,
        third_path=third_path,
        results=results,
        history_results=previous_history,
        target_p50_sec=args.target_p50_sec,
        target_p90_sec=args.target_p90_sec,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark summary: {out_path}", flush=True)
    print(f"Saved latest benchmark alias: {latest_path}", flush=True)
    print(json.dumps(summary["client_end_to_end_sec"], ensure_ascii=False, indent=2), flush=True)
    print(f"Run history samples: {summary['history_ready_count']}/{summary['history_count']}", flush=True)


if __name__ == "__main__":
    main()

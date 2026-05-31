from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


MATRIX_FIELDS = [
    "task_type",
    "sample_id",
    "model_name",
    "input_path",
    "expected",
    "actual",
    "pass_fail",
    "response_time_ms",
    "notes",
]


def ensure_reports_dir() -> Path:
    path = PROJECT_ROOT / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_matrix_rows(path: Path, rows: Iterable[Dict[str, Any]], *, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_FIELDS)
        if not append or not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in MATRIX_FIELDS})


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            pass
    return {}


def dashscope_base_url() -> str:
    return (
        os.getenv("DASHSCOPE_BASE_URL")
        or os.getenv("DASHSCOPE_API_BASE_URL")
        or "https://dashscope.aliyuncs.com/api/v1"
    ).replace("/compatible-mode/v1", "/api/v1")


def _local_path_from_media_ref(value: str) -> Optional[Path]:
    if not value:
        return None
    if value.startswith("file://"):
        return Path(value[7:])
    if value.startswith("http://") or value.startswith("https://"):
        return None
    return Path(value)


def _compress_video_for_dashscope(
    video_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    max_width: int = 640,
    max_duration_sec: float = 8.0,
    min_duration_sec: float = 3.0,
    fps_limit: float = 6.0,
) -> Path:
    """Create a small, deterministic MP4 for DashScope video upload retries."""
    import cv2

    cache_dir = cache_dir or (PROJECT_ROOT / "reports" / "dashscope_video_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{video_path.stem}_ds_{max_width}w_{int(fps_limit)}fps_min{int(min_duration_sec)}.mp4"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return video_path
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return video_path
        scale = min(1.0, max_width / float(width))
        out_size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
        out_fps = max(1.0, min(float(fps_limit), source_fps))
        frame_step = max(1, int(round(source_fps / out_fps)))
        max_frames = int(max_duration_sec * source_fps) if max_duration_sec > 0 else frame_count
        if frame_count > 0:
            max_frames = min(max_frames, frame_count)

        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, out_size)
        if not writer.isOpened():
            return video_path
        written = 0
        last_frame = None
        idx = 0
        ok, frame = capture.read()
        while ok and frame is not None and (max_frames <= 0 or idx < max_frames):
            if idx % frame_step == 0:
                if scale < 1.0:
                    frame = cv2.resize(frame, out_size)
                writer.write(frame)
                last_frame = frame
                written += 1
            idx += 1
            ok, frame = capture.read()
        min_frames = int(max(0.0, min_duration_sec) * out_fps)
        while last_frame is not None and written < min_frames:
            writer.write(last_frame)
            written += 1
        writer.release()
        if written <= 0 or not output_path.exists() or output_path.stat().st_size <= 0:
            return video_path
        return output_path
    finally:
        capture.release()


def _prepare_multimodal_content(
    content: List[Dict[str, Any]],
    *,
    compress_video: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prepared: List[Dict[str, Any]] = []
    media_records: List[Dict[str, Any]] = []
    for item in content:
        if compress_video and "video" in item:
            original_ref = str(item.get("video") or "")
            local_path = _local_path_from_media_ref(original_ref)
            if local_path and local_path.exists():
                compressed = _compress_video_for_dashscope(local_path)
                prepared.append({**item, "video": file_uri(compressed)})
                media_records.append(
                    {
                        "type": "video",
                        "original": str(local_path),
                        "prepared": str(compressed),
                        "original_size_bytes": local_path.stat().st_size,
                        "prepared_size_bytes": compressed.stat().st_size if compressed.exists() else None,
                    }
                )
                continue
        prepared.append(item)
    return prepared, media_records


def _record_dashscope_failure(
    *,
    model: str,
    content: List[Dict[str, Any]],
    error: BaseException,
    attempt: int,
    response: Optional[Dict[str, Any]] = None,
) -> Path:
    failed_dir = PROJECT_ROOT / "reports" / "dashscope_failures"
    failed_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    failure_path = failed_dir / f"{stamp}_{model}_attempt{attempt}.json"
    payload = {
        "model": model,
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": str(error),
        "content": content,
        "response": response or {},
    }
    failure_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in content:
        ref = str(item.get("video") or item.get("image") or "")
        local_path = _local_path_from_media_ref(ref)
        if local_path and local_path.exists():
            try:
                shutil.copy2(local_path, failed_dir / local_path.name)
            except Exception:
                pass
    return failure_path


def dashscope_call_multimodal(
    *,
    model: str,
    content: List[Dict[str, Any]],
    temperature: float = 0.1,
    timeout: int = 90,
    retries: int = 2,
    compress_video: bool = False,
) -> Dict[str, Any]:
    import dashscope  # type: ignore

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")
    dashscope.base_http_api_url = dashscope_base_url()
    prepared_content, media_records = _prepare_multimodal_content(content, compress_video=compress_video)
    last_error: Optional[BaseException] = None
    last_response: Dict[str, Any] = {}
    max_attempts = max(1, int(retries) + 1)
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            response = dashscope.MultiModalConversation.call(
                api_key=api_key,
                model=model,
                messages=[{"role": "user", "content": prepared_content}],
                temperature=temperature,
                result_format="message",
                timeout=timeout,
            )
            response_time_ms = int((time.perf_counter() - started) * 1000)
            text = extract_dashscope_text(response)
            response_code = getattr(response, "code", None) if not isinstance(response, dict) else response.get("code")
            response_message = getattr(response, "message", None) if not isinstance(response, dict) else response.get("message")
            last_response = {
                "response_code": response_code,
                "response_message": response_message,
                "raw_response": text,
            }
            if text:
                return {
                    "response_time_ms": response_time_ms,
                    "raw_response": text,
                    "response_code": response_code,
                    "response_message": response_message,
                    "base_url": dashscope.base_http_api_url,
                    "attempts": attempt,
                    "prepared_media": media_records,
                }
            last_error = RuntimeError(f"DashScope multimodal response missing text: {response_code} {response_message}")
            failure_path = _record_dashscope_failure(
                model=model,
                content=prepared_content,
                error=last_error,
                attempt=attempt,
                response=last_response,
            )
            last_response["failure_record_path"] = str(failure_path)
        except Exception as exc:
            last_error = exc
            failure_path = _record_dashscope_failure(model=model, content=prepared_content, error=exc, attempt=attempt)
            last_response = {"failure_record_path": str(failure_path)}
        if attempt < max_attempts:
            time.sleep(min(2 * attempt, 6))
    assert last_error is not None
    raise RuntimeError(
        f"DashScope multimodal call failed after {max_attempts} attempt(s): {last_error}; "
        f"failure_record={last_response.get('failure_record_path')}"
    ) from last_error


def extract_dashscope_text(response: Any) -> str:
    output = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
    choices = (output or {}).get("choices") if isinstance(output, dict) else []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or []
    if isinstance(content, str):
        return content.strip()
    parts: List[str] = []
    for item in content:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def file_uri(path: Path) -> str:
    return f"file://{path.resolve().as_posix()}"


def compact(value: Any, limit: int = 600) -> str:
    text = _csv_value(value)
    return text[:limit]


def model_list(default: str = "qwen3.6-plus,qwen3.6-flash") -> List[str]:
    return [item.strip() for item in os.getenv("QWEN_VISION_EVAL_MODELS", default).split(",") if item.strip()]


def pass_fail_from_structured(payload: Dict[str, Any], required: List[str]) -> tuple[str, str]:
    if not payload:
        return "fail", "no structured JSON object parsed"
    missing = [key for key in required if key not in payload]
    if missing:
        return "fail", "missing keys: " + ",".join(missing)
    return "pass", "structured keys present"

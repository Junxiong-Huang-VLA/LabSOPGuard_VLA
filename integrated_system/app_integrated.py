from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import os
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from integrated_system.config import IntegratedSettings, load_settings
from integrated_system.hand_detection import run_hand_detection
from integrated_system.integrated_pdf_generator import generate_integrated_pdf
from integrated_system.keyframe_ai import run_keyframe_ai_pipeline
from integrated_system.step_checker import run_step_check


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES_YAML = PROJECT_ROOT / "configs" / "sop" / "rules.yaml"


TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()
PROGRESS_EVENTS: list[Dict[str, Any]] = []
PROGRESS_COND = threading.Condition()
TASKS_PERSIST_LOCK = threading.Lock()
TASKS_BOOTSTRAP_LOCK = threading.Lock()
TASKS_BOOTSTRAPPED = False
PERSIST_META: Dict[str, Any] = {"last_saved_at": "", "last_error": ""}
PERSIST_INTERVAL_SEC = 1.0

SETTINGS = load_settings()
EXECUTOR = ThreadPoolExecutor(max_workers=SETTINGS.max_workers, thread_name_prefix="integrated-worker")
TASK_REGISTRY_FILE = SETTINGS.outputs_root / SETTINGS.task_registry_filename

VALID_TASK_STATUS = {"pending", "running", "completed", "failed"}


class TaskCancelled(Exception):
    pass


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_task_status(status: Any, default: str = "pending") -> str:
    s = str(status or "").strip().lower()
    if s in VALID_TASK_STATUS:
        return s
    return default


def _clamp_progress(progress: Any) -> float:
    try:
        p = float(progress)
    except Exception:
        p = 0.0
    return max(0.0, min(100.0, p))


def _safe_message(message: Any, default: str = "") -> str:
    msg = str(message or "").strip()
    return msg or default


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _parse_utc_iso(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _json_safe_clone(payload: Any) -> Any:
    try:
        return json.loads(json.dumps(payload, ensure_ascii=False))
    except Exception:
        return payload


def _trim_task_registry(max_items: int = 600) -> None:
    with TASKS_LOCK:
        if len(TASKS) <= max_items:
            return
        ordered = sorted(
            TASKS.items(),
            key=lambda x: str(x[1].get("created_at", "")),
            reverse=True,
        )
        keep = dict(ordered[:max_items])
        TASKS.clear()
        TASKS.update(keep)


def _persist_tasks_to_disk(force: bool = False) -> None:
    now_ts = time.time()
    last_saved_ts = float(PERSIST_META.get("last_saved_ts") or 0.0)
    if not force and (now_ts - last_saved_ts) < PERSIST_INTERVAL_SEC:
        return

    with TASKS_PERSIST_LOCK:
        last_saved_ts = float(PERSIST_META.get("last_saved_ts") or 0.0)
        if not force and (now_ts - last_saved_ts) < PERSIST_INTERVAL_SEC:
            return

        try:
            SETTINGS.outputs_root.mkdir(parents=True, exist_ok=True)
            with TASKS_LOCK:
                tasks_snapshot = _json_safe_clone(TASKS)
            payload = {
                "version": 1,
                "saved_at": _utc_now_iso(),
                "task_count": len(tasks_snapshot),
                "tasks": tasks_snapshot,
            }
            tmp_path = TASK_REGISTRY_FILE.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(TASK_REGISTRY_FILE)
            PERSIST_META["last_saved_at"] = str(payload["saved_at"])
            PERSIST_META["last_saved_ts"] = now_ts
            PERSIST_META["last_error"] = ""
            PERSIST_META["registry_path"] = _safe_rel(TASK_REGISTRY_FILE)
        except Exception as exc:
            PERSIST_META["last_error"] = str(exc)


def _normalize_loaded_task(task_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
    task_norm = dict(task) if isinstance(task, dict) else {}
    task_norm["task_id"] = task_id
    task_norm["status"] = _normalize_task_status(task_norm.get("status"), default="failed")
    task_norm["progress"] = _clamp_progress(task_norm.get("progress", 0.0))
    task_norm["current_stage"] = _safe_message(task_norm.get("current_stage"), "等待执行")
    task_norm["message"] = _safe_message(task_norm.get("message"), "")
    task_norm["updated_at"] = _safe_message(task_norm.get("updated_at"), _utc_now_iso())
    task_norm["created_at"] = _safe_message(task_norm.get("created_at"), task_norm["updated_at"])
    task_norm["outputs"] = dict(task_norm.get("outputs", {}) if isinstance(task_norm.get("outputs"), dict) else {})
    task_norm["options"] = dict(task_norm.get("options", {}) if isinstance(task_norm.get("options"), dict) else {})
    task_norm["module_status"] = dict(
        task_norm.get("module_status", {}) if isinstance(task_norm.get("module_status"), dict) else {}
    )
    task_norm["module_notes"] = list(task_norm.get("module_notes", []) if isinstance(task_norm.get("module_notes"), list) else [])
    return task_norm


def _bootstrap_tasks_from_disk() -> int:
    global TASKS_BOOTSTRAPPED
    with TASKS_BOOTSTRAP_LOCK:
        if TASKS_BOOTSTRAPPED:
            return len(TASKS)

        restored: Dict[str, Dict[str, Any]] = {}
        if TASK_REGISTRY_FILE.exists():
            try:
                raw = json.loads(TASK_REGISTRY_FILE.read_text(encoding="utf-8"))
                source = raw.get("tasks", {}) if isinstance(raw, dict) else {}
                if isinstance(source, dict):
                    for task_id, task_payload in source.items():
                        tid = str(task_id or "").strip()
                        if not tid:
                            continue
                        if not isinstance(task_payload, dict):
                            continue
                        norm_task = _normalize_loaded_task(tid, task_payload)
                        # 服务重启后，运行中任务转失败，避免“假运行”状态卡死。
                        if norm_task["status"] in {"pending", "running"}:
                            norm_task["status"] = "failed"
                            norm_task["progress"] = 100.0
                            norm_task["current_stage"] = "失败"
                            norm_task["message"] = "服务已重启，任务在重启前中断。请重试。"
                            norm_task["updated_at"] = _utc_now_iso()
                            module_status = dict(norm_task.get("module_status", {}))
                            module_status["pipeline"] = {
                                "status": "failed",
                                "message": "服务重启导致任务中断",
                                "updated_at": norm_task["updated_at"],
                            }
                            norm_task["module_status"] = module_status
                        restored[tid] = norm_task
            except Exception as exc:
                PERSIST_META["last_error"] = f"load_registry_failed: {exc}"

        with TASKS_LOCK:
            TASKS.clear()
            TASKS.update(restored)
        _trim_task_registry()
        TASKS_BOOTSTRAPPED = True
        _persist_tasks_to_disk(force=True)
        return len(TASKS)


def _is_task_cancel_requested(task_id: str) -> bool:
    snap = _task_snapshot(task_id)
    return bool(snap.get("cancel_requested"))


def _ensure_not_cancelled(task_id: str) -> None:
    if _is_task_cancel_requested(task_id):
        raise TaskCancelled("任务已取消")


def _progress_ratio_to_range(ratio: float, start: float, end: float) -> float:
    r = max(0.0, min(1.0, float(ratio)))
    return start + (end - start) * r


def _resolve_project_file(rel: str) -> Optional[Path]:
    rel_norm = str(rel or "").replace("\\", "/").strip().lstrip("/")
    if not rel_norm:
        return None
    candidate = (PROJECT_ROOT / rel_norm).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except Exception:
        return None
    return candidate


def _set_task_module_status(
    task_id: str,
    module_name: str,
    status: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    status_norm = str(status).strip().lower()
    if status_norm not in {"completed", "skipped", "failed"}:
        status_norm = "failed"

    with TASKS_LOCK:
        task = TASKS.get(task_id, {})
        module_status = dict(task.get("module_status", {}) if isinstance(task.get("module_status"), dict) else {})
        payload: Dict[str, Any] = {
            "status": status_norm,
            "message": _safe_message(message, "no details"),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        if details:
            payload["details"] = details
        module_status[module_name] = payload
        task["module_status"] = module_status
        task["updated_at"] = datetime.utcnow().isoformat() + "Z"
        TASKS[task_id] = task


def _set_task_runtime_state(
    task_id: str,
    *,
    status: str,
    progress: float,
    stage: str,
    message: str,
    emit_event: bool = True,
) -> None:
    status_norm = _normalize_task_status(status, default="running")
    stage_text = _safe_message(stage, "处理中")
    message_text = _safe_message(message, "处理中")
    progress_value = _clamp_progress(progress)
    with TASKS_LOCK:
        task = TASKS.get(task_id, {})
        prev_progress = _clamp_progress(task.get("progress", 0.0))
        if status_norm in {"pending", "running"}:
            progress_value = max(prev_progress, progress_value)
        task.update(
            {
                "status": status_norm,
                "progress": progress_value,
                "current_stage": stage_text,
                "message": message_text,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        TASKS[task_id] = task
    if emit_event:
        _emit_progress(task_id, progress_value, stage_text, message_text, status=status_norm)


def _load_rules_config() -> dict:
    if not RULES_YAML.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_rules_expected_steps() -> list[str]:
    cfg = _load_rules_config()
    raw = cfg.get("required_steps") if isinstance(cfg, dict) else []
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]
    return [
        "wear_ppe",
        "verify_reagent_label",
        "prepare_transfer_tool",
        "execute_transfer",
        "close_container",
        "clean_workspace",
        "dispose_waste",
    ]


def _emit_progress(task_id: str, progress: float, stage: str, message: str, status: Optional[str] = None) -> None:
    status_value = _normalize_task_status(status, default="running")
    event = {
        "task_id": task_id,
        "status": status_value,
        "progress": _clamp_progress(progress),
        "current_stage": _safe_message(stage, "处理中"),
        "message": _safe_message(message, "处理中"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with PROGRESS_COND:
        PROGRESS_EVENTS.append(event)
        if len(PROGRESS_EVENTS) > 5000:
            del PROGRESS_EVENTS[:1000]
        PROGRESS_COND.notify_all()


def _update_task(task_id: str, **kwargs: Any) -> None:
    with TASKS_LOCK:
        task = TASKS.get(task_id, {})
        task.update(kwargs)
        task["updated_at"] = datetime.utcnow().isoformat() + "Z"
        TASKS[task_id] = task


def _update_task_outputs(task_id: str, outputs: Dict[str, str]) -> None:
    _update_task(task_id, outputs=dict(outputs))


def _task_snapshot(task_id: str) -> Dict[str, Any]:
    with TASKS_LOCK:
        t = dict(TASKS.get(task_id, {}))
    return t


def _safe_rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _safe_read_text(path: Path, limit: int = 8000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    raw = raw.strip()
    if len(raw) > limit:
        return raw[:limit] + "\n...[truncated]"
    return raw


def _task_public_payload(task_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
    output_dir_rel = str(task.get("output_dir", ""))
    output_dir = _resolve_project_file(output_dir_rel) if output_dir_rel else None
    outputs = dict(task.get("outputs", {}) if isinstance(task.get("outputs"), dict) else {})
    module_status = dict(task.get("module_status", {}) if isinstance(task.get("module_status"), dict) else {})
    module_notes = list(task.get("module_notes", []) if isinstance(task.get("module_notes"), list) else [])

    keyframe_images: List[str] = []
    keyframe_items: List[Dict[str, Any]] = []
    part1_rel = str(outputs.get("part1_keyframes") or "")
    part1_abs = _resolve_project_file(part1_rel) if part1_rel else None
    if not part1_abs and output_dir:
        maybe_part1 = output_dir / "part1_keyframes.json"
        part1_abs = maybe_part1 if maybe_part1.exists() else None

    if part1_abs and part1_abs.exists():
        try:
            part1_data = json.loads(part1_abs.read_text(encoding="utf-8"))
            for item in part1_data.get("keyframes", []):
                img_name = str(item.get("image", "")).strip()
                if not img_name:
                    continue
                rel = f"keyframes/{img_name}"
                keyframe_images.append(rel)
                keyframe_items.append(
                    {
                        "image": img_name,
                        "timestamp": item.get("timestamp"),
                        "frame_id": item.get("frame_id"),
                        "url": f"/api/artifact/{task_id}?path={rel}",
                    }
                )
        except Exception:
            pass

    if output_dir and output_dir.exists() and not keyframe_images:
        for p in sorted((output_dir / "keyframes").glob("keyframe_*.jpg"))[:36]:
            rel = f"keyframes/{p.name}"
            keyframe_images.append(rel)
            keyframe_items.append({"image": p.name, "timestamp": None, "frame_id": None, "url": f"/api/artifact/{task_id}?path={rel}"})

    keyframe_urls = [f"/api/artifact/{task_id}?path={rel}" for rel in keyframe_images]

    annotated_rel = str(outputs.get("annotated_video") or "")
    annotated_abs = _resolve_project_file(annotated_rel) if annotated_rel else None
    annotated_exists = bool(annotated_abs and annotated_abs.exists() and annotated_abs.is_file())
    annotated_artifact_path = ""
    if annotated_exists and output_dir and annotated_abs is not None:
        try:
            annotated_artifact_path = str(annotated_abs.relative_to(output_dir)).replace("\\", "/")
        except Exception:
            annotated_artifact_path = "hand_annotated.mp4"

    summary_rel = str(outputs.get("overall_summary") or "")
    summary_abs = _resolve_project_file(summary_rel) if summary_rel else None
    if not summary_abs and output_dir:
        maybe_summary = output_dir / "overall_summary.txt"
        summary_abs = maybe_summary if maybe_summary.exists() else None
    summary_text = _safe_read_text(summary_abs) if summary_abs else ""
    summary_excerpt = summary_text[:700] + ("..." if len(summary_text) > 700 else "")

    alarm_rel = str(outputs.get("alarm_log") or "")
    alarm_abs = _resolve_project_file(alarm_rel) if alarm_rel else None
    if not alarm_abs and output_dir:
        maybe_alarm = output_dir / "alarm_log.json"
        alarm_abs = maybe_alarm if maybe_alarm.exists() else None
    alarm_exists = bool(alarm_abs and alarm_abs.exists() and alarm_abs.is_file())
    alarm_count = 0
    alarm_preview: List[str] = []
    if alarm_exists and alarm_abs:
        try:
            alarm_payload = json.loads(alarm_abs.read_text(encoding="utf-8"))
            alarms = alarm_payload.get("alarms", []) if isinstance(alarm_payload, dict) else []
            alarm_count = int(alarm_payload.get("alarm_count", len(alarms))) if isinstance(alarm_payload, dict) else len(alarms)
            for a in alarms[:6]:
                desc = _safe_message(a.get("description", "无描述"), "无描述")
                typ = _safe_message(a.get("alarm_type", "unknown"), "unknown")
                sev = _safe_message(a.get("severity", "unknown"), "unknown")
                ts = a.get("timestamp", "na")
                alarm_preview.append(f"[{sev}] {typ} @ {ts}: {desc}")
        except Exception:
            alarm_count = 0
            alarm_preview = []

    report_rel = str(outputs.get("report") or "")
    report_abs = _resolve_project_file(report_rel) if report_rel else None
    if not report_abs and output_dir:
        maybe_report_pdf = output_dir / "integrated_analysis_report.pdf"
        maybe_report_txt = output_dir / "integrated_analysis_report.txt"
        if maybe_report_pdf.exists():
            report_abs = maybe_report_pdf
        elif maybe_report_txt.exists():
            report_abs = maybe_report_txt
    report_exists = bool(report_abs and report_abs.exists() and report_abs.is_file())
    report_is_pdf = bool(report_exists and report_abs and report_abs.suffix.lower() == ".pdf")
    report_is_txt = bool(report_exists and report_abs and report_abs.suffix.lower() == ".txt")

    analysis_rel = str(outputs.get("analysis_json") or "")
    analysis_abs = _resolve_project_file(analysis_rel) if analysis_rel else None
    if not analysis_abs and output_dir:
        maybe_analysis = output_dir / "keyframe_ai_analysis.json"
        analysis_abs = maybe_analysis if maybe_analysis.exists() else None
    analysis_exists = bool(analysis_abs and analysis_abs.exists() and analysis_abs.is_file())

    part1_exists = bool(part1_abs and part1_abs.exists() and part1_abs.is_file())
    summary_exists = bool(summary_abs and summary_abs.exists() and summary_abs.is_file())

    download_ready = {
        "annotated_video": annotated_exists,
        "summary": summary_exists,
        "pdf": report_is_pdf,
        "alarm_log": alarm_exists,
        "keyframe_json": part1_exists,
        "analysis_json": analysis_exists,
    }

    public = {
        "task_id": task_id,
        "status": _normalize_task_status(task.get("status"), default="pending"),
        "progress": _clamp_progress(task.get("progress", 0.0)),
        "current_stage": _safe_message(task.get("current_stage", "等待执行"), "等待执行"),
        "message": _safe_message(task.get("message", ""), ""),
        "outputs": outputs,
        "module_status": module_status,
        "module_notes": module_notes,
        "output_dir": output_dir_rel,
        "video_path": str(task.get("video_path", "")),
        "upload": task.get("upload", {}),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "options": task.get("options", {}),
        "error": task.get("error"),
        "failure": _classify_failure(task),
        "artifacts": {
            "annotated_video_url": (
                f"/api/artifact/{task_id}?path={annotated_artifact_path}" if (annotated_exists and annotated_artifact_path) else ""
            ),
            "annotated_video_exists": annotated_exists,
            "keyframe_image_urls": keyframe_urls,
            "keyframe_items": keyframe_items,
            "keyframe_count": len(keyframe_urls),
            "summary_text": summary_text,
            "summary_excerpt": summary_excerpt,
            "alarm_log_exists": alarm_exists,
            "alarm_count": alarm_count,
            "alarm_preview": alarm_preview,
            "pdf_exists": report_exists,
            "pdf_generated": report_is_pdf,
            "pdf_fallback_txt": report_is_txt,
            "analysis_exists": analysis_exists,
            "download_ready": download_ready,
            "report_path": report_rel,
        },
    }
    return public


def _classify_failure(task: Dict[str, Any]) -> Optional[Dict[str, str]]:
    status = _normalize_task_status(task.get("status"), default="")
    if status != "failed":
        return None

    stage = _safe_message(task.get("current_stage", ""), "unknown")
    message = _safe_message(task.get("message", ""), "")
    error = _safe_message(task.get("error", ""), "")
    blob = f"{message}\n{error}".lower()

    category = "runtime_error"
    level = "P2"
    title = "Runtime Failure"
    hint = "Inspect traceback and stage logs, then retry the task."

    if "permissionerror" in blob or "access is denied" in blob or "拒绝访问" in blob:
        category = "permission_error"
        level = "P0"
        title = "Permission Denied"
        hint = "Check folder ACL and ensure the process can read/write input and output paths."
    elif "no module named" in blob or "modulenotfounderror" in blob or "importerror" in blob:
        category = "dependency_missing"
        level = "P0"
        title = "Missing Dependency"
        hint = "Install missing Python packages in the active runtime environment."
    elif "cannot open video" in blob or "视频解码" in blob or "video decode" in blob:
        category = "video_decode_error"
        level = "P1"
        title = "Video Decode Failure"
        hint = "Ensure the uploaded video is valid and codec is supported by OpenCV runtime."
    elif "api key" in blob or "auth" in blob or "unauthorized" in blob or "forbidden" in blob:
        category = "auth_or_api_error"
        level = "P1"
        title = "Auth/API Configuration Error"
        hint = "Check API key variables and endpoint configuration before retry."
    elif "ffmpeg" in blob:
        category = "ffmpeg_unavailable"
        level = "P1"
        title = "FFmpeg Unavailable"
        hint = "Install ffmpeg or keep using the OpenCV fallback export path."
    elif "file not found" in blob or "no such file" in blob or "missing video" in blob:
        category = "input_or_file_error"
        level = "P1"
        title = "Input/File Error"
        hint = "Verify uploaded video and intermediate files exist and are readable."
    elif "timeout" in blob or "timed out" in blob:
        category = "timeout"
        level = "P1"
        title = "Timeout"
        hint = "Reduce workload or increase timeout/resource limits and retry."

    # Stage-specific hints make triage faster for operators.
    stage_norm = stage.strip().lower()
    if stage_norm in {"hand_detection", "手部检测中"} and category == "runtime_error":
        title = "Hand Detection Failure"
        hint = "Check mediapipe/opencv runtime and video codec compatibility."
    elif stage_norm in {"keyframe_extract", "关键帧提取中", "ai分析中"} and category == "runtime_error":
        title = "Keyframe/AI Analysis Failure"
        hint = "Check keyframe extraction parameters and AI model endpoint availability."
    elif stage_norm in {"step_check", "步骤检查中"} and category == "runtime_error":
        title = "SOP Step Check Failure"
        hint = "Validate SOP rules YAML and keyframe analysis payload format."
    elif stage_norm in {"pdf", "pdf生成中"} and category == "runtime_error":
        title = "PDF Generation Failure"
        hint = "Check reportlab/font setup; fallback TXT should still be available."

    return {
        "category": category,
        "level": level,
        "stage": stage,
        "title": title,
        "hint": hint,
        "message": message,
    }


def _is_openai_key_configured() -> bool:
    return bool(os.getenv("DOUBAO_API_KEY") or os.getenv("ARK_API_KEY"))


def _resolve_ffmpeg_executable() -> Optional[str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore

        alt = get_ffmpeg_exe()
        if alt and Path(alt).exists():
            return str(alt)
    except Exception:
        return None
    return None


def _probe_modules() -> Dict[str, bool]:
    modules = {}
    for mod_name in ("cv2", "mediapipe", "openai", "reportlab"):
        try:
            __import__(mod_name)
            modules[mod_name] = True
        except Exception:
            modules[mod_name] = False
    return modules


def _run_pipeline(task_id: str, video_path: Path, output_dir: Path, options: Dict[str, Any], settings: IntegratedSettings) -> None:
    outputs: Dict[str, str] = {}
    hand_payload: Dict[str, Any] = {}
    keyframe_payload: Dict[str, Any] = {}
    alarm_payload: Dict[str, Any] = {}
    module_notes: List[str] = []

    def _append_module_note(note: str) -> None:
        note_text = _safe_message(note, "")
        if not note_text:
            return
        if note_text not in module_notes:
            module_notes.append(note_text)
            _update_task(task_id, module_notes=list(module_notes))

    def _human_ai_skip_reason(reason: str) -> str:
        reason_norm = str(reason or "").strip().lower()
        if reason_norm == "disabled_by_option":
            return "AI分析已跳过：开关关闭。"
        if reason_norm == "missing_api_key":
            return "AI分析已跳过：未配置 DOUBAO_API_KEY / ARK_API_KEY。"
        if reason_norm == "openai_package_missing":
            return "AI分析已跳过：未安装 openai 依赖。"
        if reason_norm == "client_init_failed":
            return "AI分析已跳过：OpenAI 客户端初始化失败。"
        if reason_norm == "no_keyframes":
            return "AI分析已跳过：未提取到关键帧。"
        return f"AI分析已跳过：{reason or 'unknown reason'}。"

    def _ensure_output_exists(key: str, fallback_path: Path) -> bool:
        rel = outputs.get(key)
        p = _resolve_project_file(rel) if rel else None
        if p and p.exists() and p.is_file():
            return True
        if fallback_path.exists() and fallback_path.is_file():
            outputs[key] = _safe_rel(fallback_path)
            return True
        return False

    def _safe_ratio(numerator: Any, denominator: Any) -> float:
        try:
            n = float(numerator)
            d = float(denominator)
        except Exception:
            return 0.0
        if d <= 0:
            return 0.0
        return max(0.0, min(1.0, n / d))

    ffmpeg_available = bool(_resolve_ffmpeg_executable())

    try:
        input_size_mb = round((video_path.stat().st_size / (1024 * 1024)), 2) if video_path.exists() else 0.0
        _set_task_module_status(
            task_id,
            "upload",
            "completed",
            f"视频上传完成：{video_path.name} ({input_size_mb} MB)",
            details={"filename": video_path.name, "size_mb": input_size_mb},
        )
        _set_task_runtime_state(
            task_id,
            status="running",
            progress=4.0,
            stage="上传完成",
            message=f"上传完成，文件 {video_path.name}（{input_size_mb} MB），开始分析。",
        )

        if options.get("enable_video_export", True) and not ffmpeg_available:
            _append_module_note("FFmpeg 不可用，标注视频将使用 OpenCV 基础导出。")

        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"视频处理依赖缺失：{exc}") from exc

        probe = cv2.VideoCapture(str(video_path))
        if not probe.isOpened():
            probe.release()
            raise RuntimeError("视频解码失败：无法打开上传视频，请检查编码格式。")
        probe.release()

        if options.get("enable_hand_detection", True):
            _set_task_runtime_state(
                task_id,
                status="running",
                progress=8.0,
                stage="手部检测中",
                message="正在执行手部检测。",
            )
            last_hand_emit = {"t": 0.0}

            def _hand_progress(processed_frames: int, total_frames: int, sampled_frames: int, msg: str) -> None:
                now = time.time()
                if now - last_hand_emit["t"] < 0.25 and total_frames > 0 and processed_frames < total_frames:
                    return
                last_hand_emit["t"] = now
                ratio = _safe_ratio(processed_frames, total_frames) if total_frames > 0 else min(sampled_frames / 200.0, 1.0)
                progress_value = _progress_ratio_to_range(ratio, 8.0, 30.0)
                detail_msg = _safe_message(msg, "手部检测中")
                _set_task_runtime_state(
                    task_id,
                    status="running",
                    progress=progress_value,
                    stage="手部检测中",
                    message=detail_msg,
                )

            try:
                hand_payload = run_hand_detection(
                    video_path=str(video_path),
                    output_dir=output_dir,
                    enable_video_export=bool(options.get("enable_video_export", True)),
                    progress_callback=_hand_progress,
                )
                outputs["hand_json"] = _safe_rel(output_dir / "hand_detection.json")
                if hand_payload.get("annotated_video"):
                    outputs["annotated_video"] = _safe_rel(Path(str(hand_payload["annotated_video"])))
                _update_task_outputs(task_id, outputs)
                hand_summary = hand_payload.get("summary", {}) if isinstance(hand_payload.get("summary"), dict) else {}
                sampled = int(hand_summary.get("total_sampled_frames", 0) or 0)
                with_hands = int(hand_summary.get("frames_with_hands", 0) or 0)

                annotated_exists = _ensure_output_exists("annotated_video", output_dir / "hand_annotated.mp4")
                if bool(options.get("enable_video_export", True)) and not annotated_exists:
                    msg = "标注视频未生成：视频导出已降级，仅保留手部检测 JSON。"
                    _set_task_module_status(task_id, "video_export", "failed", msg)
                    _append_module_note(msg)
                elif bool(options.get("enable_video_export", True)):
                    _set_task_module_status(task_id, "video_export", "completed", "标注视频生成完成。")
                else:
                    _set_task_module_status(task_id, "video_export", "skipped", "标注视频导出开关关闭。")

                warnings = hand_summary.get("warnings", []) if isinstance(hand_summary.get("warnings"), list) else []
                for w in warnings:
                    _append_module_note(f"手部检测提示：{w}")

                mediapipe_enabled = bool(hand_summary.get("mediapipe_enabled", False))
                if mediapipe_enabled:
                    hand_msg = f"手部检测完成：采样 {sampled} 帧，检出手部 {with_hands} 帧。"
                    hand_status = "completed"
                else:
                    hand_msg = f"手部检测已跳过：mediapipe 不可用（已采样 {sampled} 帧，检出手部 {with_hands} 帧）。"
                    hand_status = "skipped"
                _set_task_module_status(
                    task_id,
                    "hand_detection",
                    hand_status,
                    hand_msg,
                )
                _set_task_runtime_state(
                    task_id,
                    status="running",
                    progress=30.0,
                    stage="手部检测中",
                    message=hand_msg,
                )
            except Exception as hand_exc:
                # Hand detection should not kill the whole pipeline.
                hand_payload = {
                    "summary": {
                        "total_sampled_frames": 0,
                        "frames_with_hands": 0,
                        "hand_presence_ratio": 0.0,
                        "max_hands_in_frame": 0,
                        "mediapipe_enabled": False,
                        "hand_backend": "failed",
                        "warnings": [f"hand detection failed: {hand_exc}"],
                    },
                    "warnings": [f"hand detection failed: {hand_exc}"],
                }
                msg = f"手部检测失败，已跳过该模块：{hand_exc}"
                _set_task_module_status(task_id, "hand_detection", "failed", msg)
                _append_module_note(msg)
                if bool(options.get("enable_video_export", True)):
                    _set_task_module_status(task_id, "video_export", "failed", "标注视频导出失败：依赖手部检测结果。")
                _set_task_runtime_state(
                    task_id,
                    status="running",
                    progress=30.0,
                    stage="手部检测中",
                    message=msg,
                )
        else:
            msg = "手部检测已跳过：开关关闭。"
            _set_task_module_status(task_id, "hand_detection", "skipped", msg)
            if bool(options.get("enable_video_export", True)):
                _set_task_module_status(task_id, "video_export", "skipped", "标注视频导出已跳过：手部检测未启用。")
            else:
                _set_task_module_status(task_id, "video_export", "skipped", "标注视频导出开关关闭。")
            _append_module_note(msg)
            _set_task_runtime_state(
                task_id,
                status="running",
                progress=30.0,
                stage="手部检测中",
                message=msg,
            )

        _set_task_runtime_state(
            task_id,
            status="running",
            progress=32.0,
            stage="关键帧提取中",
            message="开始提取关键帧。",
        )

        last_extract_emit = {"t": 0.0}

        def _extract_progress(processed: int, total: int, saved: int, msg: str) -> None:
            now = time.time()
            if now - last_extract_emit["t"] < 0.25 and total > 0 and processed < total:
                return
            last_extract_emit["t"] = now
            ratio = _safe_ratio(processed, total) if total > 0 else min(saved / max(float(settings.max_keyframes), 1.0), 1.0)
            progress_value = _progress_ratio_to_range(ratio, 32.0, 58.0)
            _set_task_runtime_state(
                task_id,
                status="running",
                progress=progress_value,
                stage="关键帧提取中",
                message=f"{_safe_message(msg, '关键帧提取中')} 当前已提取 {saved} 帧。",
            )

        last_ai_emit = {"t": 0.0}

        def _ai_progress(done: int, total: int, msg: str) -> None:
            now = time.time()
            if now - last_ai_emit["t"] < 0.2 and total > 0 and done < total:
                return
            last_ai_emit["t"] = now
            ratio = _safe_ratio(done, total) if total > 0 else (1.0 if done > 0 else 0.0)
            progress_value = _progress_ratio_to_range(ratio, 58.0, 78.0)
            _set_task_runtime_state(
                task_id,
                status="running",
                progress=progress_value,
                stage="AI分析中",
                message=_safe_message(msg, "AI分析中"),
            )

        keyframe_payload = run_keyframe_ai_pipeline(
            video_path=str(video_path),
            output_dir=output_dir,
            diff_threshold=settings.keyframe_diff_threshold,
            min_interval_sec=settings.keyframe_min_interval_sec,
            max_keyframes=settings.max_keyframes,
            enable_ai_analysis=bool(options.get("enable_ai_analysis", True)),
            ai_model=settings.ai_model,
            ai_base_url=settings.ai_base_url,
            extraction_progress_callback=_extract_progress,
            ai_progress_callback=_ai_progress,
        )
        outputs["part1_keyframes"] = _safe_rel(output_dir / "part1_keyframes.json")
        outputs["overall_summary"] = _safe_rel(output_dir / "overall_summary.txt")
        outputs["analysis_json"] = _safe_rel(output_dir / "keyframe_ai_analysis.json")
        _update_task_outputs(task_id, outputs)

        keyframes = keyframe_payload.get("keyframes", []) if isinstance(keyframe_payload.get("keyframes"), list) else []
        keyframe_count = len(keyframes)
        if keyframe_count <= 0:
            msg = "关键帧提取结果为空：未提取到关键帧。"
            _set_task_module_status(task_id, "keyframe_extract", "failed", msg)
            _append_module_note(msg)
        else:
            _set_task_module_status(task_id, "keyframe_extract", "completed", f"关键帧提取完成：共 {keyframe_count} 帧。")

        ai_result = keyframe_payload.get("analysis", {}) if isinstance(keyframe_payload.get("analysis"), dict) else {}
        ai_enabled = bool(ai_result.get("enabled", False))
        ai_reason = str(ai_result.get("reason", "") or "")
        local_ai_fallback = ai_reason.startswith("local_fallback_")
        ai_analyses = ai_result.get("analyses", []) if isinstance(ai_result.get("analyses"), list) else []
        ai_failed_calls = 0
        for item in ai_analyses:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary", "") or "").lower()
            if "ai call failed:" in summary:
                ai_failed_calls += 1

        if bool(options.get("enable_ai_analysis", True)):
            if ai_enabled and ai_analyses:
                if local_ai_fallback:
                    msg = f"AI分析完成：云端不可用，已使用本地视觉回退（共处理 {len(ai_analyses)} 帧）。"
                    _set_task_module_status(task_id, "ai_analysis", "completed", msg)
                    _append_module_note("AI云端不可用，已自动切换本地视觉分析。")
                elif ai_failed_calls == len(ai_analyses):
                    msg = "AI分析执行失败：模型调用均失败。"
                    _set_task_module_status(task_id, "ai_analysis", "failed", msg)
                    _append_module_note(msg)
                elif ai_failed_calls > 0:
                    msg = f"AI分析部分失败：{ai_failed_calls}/{len(ai_analyses)} 帧调用失败。"
                    _set_task_module_status(task_id, "ai_analysis", "completed", msg)
                    _append_module_note(msg)
                else:
                    _set_task_module_status(task_id, "ai_analysis", "completed", f"AI分析完成：共处理 {len(ai_analyses)} 帧。")
            else:
                msg = _human_ai_skip_reason(ai_reason)
                _set_task_module_status(task_id, "ai_analysis", "skipped", msg)
                _append_module_note(msg)
        else:
            msg = "AI分析已跳过：开关关闭。"
            _set_task_module_status(task_id, "ai_analysis", "skipped", msg)
            _append_module_note(msg)

        if options.get("enable_step_check", True):
            _set_task_runtime_state(
                task_id,
                status="running",
                progress=80.0,
                stage="步骤检查中",
                message="正在执行 SOP 步骤检查。",
            )
            rules_cfg = _load_rules_config()
            expected = _read_rules_expected_steps()
            try:
                alarm_payload = run_step_check(
                    output_dir=output_dir,
                    keyframe_meta=keyframe_payload.get("keyframes", []),
                    keyframe_analysis=keyframe_payload.get("analysis", {}).get("analyses", []),
                    expected_steps=expected,
                    rules=rules_cfg,
                    hand_summary=hand_payload.get("summary", {}),
                )
                outputs["alarm_log"] = _safe_rel(output_dir / "alarm_log.json")
                _update_task_outputs(task_id, outputs)
                alarm_count = int(alarm_payload.get("alarm_count", len(alarm_payload.get("alarms", []))) or 0)
                _set_task_module_status(task_id, "step_check", "completed", f"步骤检查完成：检测到 {alarm_count} 条报警。")
                _set_task_runtime_state(
                    task_id,
                    status="running",
                    progress=87.0,
                    stage="步骤检查中",
                    message=f"步骤检查完成：检测到 {alarm_count} 条报警。",
                )
            except Exception as step_exc:
                msg = f"步骤检查失败，已跳过：{step_exc}"
                _set_task_module_status(task_id, "step_check", "failed", msg)
                _append_module_note(msg)
                _set_task_runtime_state(
                    task_id,
                    status="running",
                    progress=87.0,
                    stage="步骤检查中",
                    message=msg,
                )
        else:
            msg = "步骤检查未启用。"
            _set_task_module_status(task_id, "step_check", "skipped", msg)
            _append_module_note(msg)
            _set_task_runtime_state(
                task_id,
                status="running",
                progress=87.0,
                stage="步骤检查中",
                message=msg,
            )

        if options.get("enable_pdf", True):
            _set_task_runtime_state(
                task_id,
                status="running",
                progress=89.0,
                stage="PDF生成中",
                message="正在生成 PDF 报告。",
            )
            try:
                report_path = generate_integrated_pdf(
                    {
                        "task_id": task_id,
                        "video_path": str(video_path),
                        "status": "completed",
                        "hand_summary": hand_payload.get("summary", {}),
                        "overall_summary": keyframe_payload.get("analysis", {}).get("overall_summary", ""),
                        "alarms": alarm_payload.get("alarms", []),
                        "outputs": outputs,
                    },
                    output_dir / "integrated_analysis_report.pdf",
                )
                outputs["report"] = _safe_rel(report_path)
                _update_task_outputs(task_id, outputs)
                report_p = Path(report_path)
                if report_p.suffix.lower() == ".pdf" and report_p.exists():
                    _set_task_module_status(task_id, "pdf", "completed", "PDF 报告生成成功。")
                    _set_task_runtime_state(
                        task_id,
                        status="running",
                        progress=97.0,
                        stage="PDF生成中",
                        message="PDF 报告生成完成。",
                    )
                else:
                    msg = "PDF 生成失败：已降级为 TXT 报告。"
                    _set_task_module_status(task_id, "pdf", "failed", msg)
                    _append_module_note(msg)
                    _set_task_runtime_state(
                        task_id,
                        status="running",
                        progress=97.0,
                        stage="PDF生成中",
                        message=msg,
                    )
            except Exception as pdf_exc:
                msg = f"PDF 生成失败：{pdf_exc}"
                _set_task_module_status(task_id, "pdf", "failed", msg)
                _append_module_note(msg)
                _set_task_runtime_state(
                    task_id,
                    status="running",
                    progress=97.0,
                    stage="PDF生成中",
                    message=msg,
                )
        else:
            msg = "PDF 报告未启用。"
            _set_task_module_status(task_id, "pdf", "skipped", msg)
            _append_module_note(msg)
            _set_task_runtime_state(
                task_id,
                status="running",
                progress=97.0,
                stage="PDF生成中",
                message=msg,
            )

        required_outputs = {
            "part1_keyframes": output_dir / "part1_keyframes.json",
            "overall_summary": output_dir / "overall_summary.txt",
            "analysis_json": output_dir / "keyframe_ai_analysis.json",
        }
        missing_required: List[str] = []
        for k, fallback in required_outputs.items():
            if not _ensure_output_exists(k, fallback):
                missing_required.append(k)
        if missing_required:
            raise RuntimeError(f"关键输出缺失：{', '.join(missing_required)}")

        _update_task(task_id, outputs=dict(outputs), module_notes=list(module_notes))
        module_status = _task_snapshot(task_id).get("module_status", {})
        degraded_modules = []
        if isinstance(module_status, dict):
            degraded_modules = [k for k, v in module_status.items() if isinstance(v, dict) and v.get("status") in {"failed", "skipped"}]
        completion_msg = (
            f"分析完成（含 {len(degraded_modules)} 个降级模块）。"
            if degraded_modules
            else "分析完成，所有主模块执行成功。"
        )
        final_payload = {
            "task_id": task_id,
            "status": "completed",
            "progress": 100.0,
            "current_stage": "已完成",
            "message": completion_msg,
            "outputs": outputs,
            "output_dir": _safe_rel(output_dir),
            "video_path": _safe_rel(video_path),
            "hand_summary": hand_payload.get("summary", {}),
            "alarm_count": len(alarm_payload.get("alarms", [])),
            "keyframe_count": len(keyframe_payload.get("keyframes", [])),
            "module_status": module_status if isinstance(module_status, dict) else {},
            "module_notes": module_notes,
        }
        (output_dir / "task_result.json").write_text(
            json.dumps(final_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs["task_result"] = _safe_rel(output_dir / "task_result.json")

        _update_task(task_id, **{k: v for k, v in final_payload.items() if k != "task_id"})
        _set_task_runtime_state(
            task_id,
            status="completed",
            progress=100.0,
            stage="已完成",
            message=completion_msg,
        )
    except Exception as exc:
        raw_error = _safe_message(str(exc), "未知错误")
        lowered = raw_error.lower()
        if "cannot open video" in lowered or "视频解码失败" in raw_error:
            final_error = f"视频解码失败：{raw_error}"
        elif "missing_api_key" in lowered:
            final_error = "AI 分析失败：缺少 DOUBAO_API_KEY/ARK_API_KEY。"
        else:
            final_error = raw_error
        try:
            _set_task_module_status(task_id, "pipeline", "failed", final_error)
        except Exception:
            pass
        _update_task(
            task_id,
            status="failed",
            progress=100.0,
            current_stage="失败",
            message=final_error,
            error=traceback.format_exc(),
            outputs=dict(outputs),
            module_notes=list(module_notes),
        )
        _emit_progress(task_id, 100.0, "失败", final_error, status="failed")


def _normalize_bool(v: Any, default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def create_app(settings: Optional[IntegratedSettings] = None) -> Flask:
    cfg = settings or SETTINGS
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))

    @app.get("/")
    def index():
        return render_template("integrated_index.html", default_port=cfg.port)

    @app.post("/api/analyze")
    def analyze():
        up = request.files.get("video")
        if up is None or not up.filename:
            return jsonify({"ok": False, "error": "Missing video file field 'video'."}), 400

        task_id = uuid.uuid4().hex[:12]
        out_dir = cfg.outputs_root / f"{_now_ts()}_{task_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(secure_filename(up.filename)).suffix or ".mp4"
        input_video = out_dir / f"input{ext}"
        up.save(str(input_video))
        size_bytes = input_video.stat().st_size if input_video.exists() else 0

        options = {
            "enable_hand_detection": _normalize_bool(request.form.get("enable_hand_detection"), cfg.enable_hand_detection),
            "enable_ai_analysis": _normalize_bool(request.form.get("enable_ai_analysis"), cfg.enable_ai_analysis),
            "enable_pdf": _normalize_bool(request.form.get("enable_pdf"), cfg.enable_pdf),
            "enable_step_check": _normalize_bool(request.form.get("enable_step_check"), cfg.enable_step_check),
            "enable_video_export": _normalize_bool(request.form.get("enable_video_export"), cfg.enable_video_export),
        }

        task = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0.0,
            "current_stage": "上传完成",
            "message": "上传成功，任务排队中。",
            "outputs": {},
            "output_dir": _safe_rel(out_dir),
            "video_path": _safe_rel(input_video),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "options": options,
            "upload": {
                "filename": up.filename,
                "saved_name": input_video.name,
                "size_bytes": size_bytes,
            },
            "module_status": {
                "upload": {
                    "status": "completed",
                    "message": f"视频上传完成：{up.filename}",
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                }
            },
            "module_notes": [],
        }
        with TASKS_LOCK:
            TASKS[task_id] = task

        EXECUTOR.submit(_run_pipeline, task_id, input_video, out_dir, options, cfg)
        _emit_progress(task_id, 0, "上传完成", "上传成功，任务排队中。", status="pending")

        return jsonify(
            {
                "ok": True,
                "task_id": task_id,
                "status": "pending",
                "progress": 0.0,
                "current_stage": "上传完成",
                "message": "上传成功，任务已受理。",
                "outputs": {},
                "output_dir": _safe_rel(out_dir),
                "upload": task["upload"],
            }
        )

    @app.post("/api/retry/<task_id>")
    def retry(task_id: str):
        source_task = _task_snapshot(task_id)
        if not source_task:
            return jsonify({"ok": False, "error": "task not found", "task_id": task_id}), 404

        source_video_rel = str(source_task.get("video_path", "")).strip()
        source_video = PROJECT_ROOT / source_video_rel if source_video_rel else None
        if not source_video_rel or source_video is None or not source_video.exists() or not source_video.is_file():
            return jsonify({"ok": False, "error": "source video not found for retry", "task_id": task_id}), 400

        prev_options = source_task.get("options", {}) if isinstance(source_task.get("options"), dict) else {}
        options = {
            "enable_hand_detection": _normalize_bool(
                request.form.get("enable_hand_detection"),
                bool(prev_options.get("enable_hand_detection", cfg.enable_hand_detection)),
            ),
            "enable_ai_analysis": _normalize_bool(
                request.form.get("enable_ai_analysis"),
                bool(prev_options.get("enable_ai_analysis", cfg.enable_ai_analysis)),
            ),
            "enable_pdf": _normalize_bool(
                request.form.get("enable_pdf"),
                bool(prev_options.get("enable_pdf", cfg.enable_pdf)),
            ),
            "enable_step_check": _normalize_bool(
                request.form.get("enable_step_check"),
                bool(prev_options.get("enable_step_check", cfg.enable_step_check)),
            ),
            "enable_video_export": _normalize_bool(
                request.form.get("enable_video_export"),
                bool(prev_options.get("enable_video_export", cfg.enable_video_export)),
            ),
        }

        new_task_id = uuid.uuid4().hex[:12]
        out_dir = cfg.outputs_root / f"{_now_ts()}_{new_task_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ext = source_video.suffix or ".mp4"
        input_video = out_dir / f"input{ext}"
        shutil.copy2(str(source_video), str(input_video))
        size_bytes = input_video.stat().st_size if input_video.exists() else 0

        task = {
            "task_id": new_task_id,
            "status": "pending",
            "progress": 0.0,
            "current_stage": "上传完成",
            "message": f"重试任务已排队（来源 {task_id}）。",
            "outputs": {},
            "output_dir": _safe_rel(out_dir),
            "video_path": _safe_rel(input_video),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "options": options,
            "retry_of": task_id,
            "upload": {
                "filename": source_video.name,
                "saved_name": input_video.name,
                "size_bytes": size_bytes,
            },
            "module_status": {
                "upload": {
                    "status": "completed",
                    "message": f"重试视频复制完成：{source_video.name}",
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                }
            },
            "module_notes": [],
        }
        with TASKS_LOCK:
            TASKS[new_task_id] = task

        EXECUTOR.submit(_run_pipeline, new_task_id, input_video, out_dir, options, cfg)
        _emit_progress(new_task_id, 0, "上传完成", f"重试任务已排队（来源 {task_id}）。", status="pending")

        return jsonify(
            {
                "ok": True,
                "task_id": new_task_id,
                "status": "pending",
                "progress": 0.0,
                "current_stage": "上传完成",
                "message": f"重试任务已受理（来源 {task_id}）。",
                "outputs": {},
                "output_dir": _safe_rel(out_dir),
                "source_task_id": task_id,
                "upload": task["upload"],
            }
        )

    @app.get("/api/status/<task_id>")
    def status(task_id: str):
        task = _task_snapshot(task_id)
        if not task:
            return jsonify(
                {
                    "ok": False,
                    "task_id": task_id,
                    "status": "failed",
                    "progress": 0.0,
                    "current_stage": "失败",
                    "message": "任务不存在",
                    "outputs": {},
                    "module_status": {},
                    "module_notes": ["任务不存在或已被清理。"],
                    "error": "task not found",
                }
            ), 404
        return jsonify({"ok": True, **_task_public_payload(task_id, task)})

    @app.get("/api/tasks")
    def list_tasks():
        raw_limit = request.args.get("limit", "20").strip()
        raw_offset = request.args.get("offset", "0").strip()
        status_filter = request.args.get("status", "").strip().lower()
        keyword = request.args.get("q", "").strip().lower()
        if status_filter and status_filter not in VALID_TASK_STATUS:
            status_filter = ""
        try:
            limit = max(1, min(100, int(raw_limit)))
        except Exception:
            limit = 20
        try:
            offset = max(0, int(raw_offset))
        except Exception:
            offset = 0

        with TASKS_LOCK:
            entries = [(tid, dict(task)) for tid, task in TASKS.items()]

        entries.sort(key=lambda x: str(x[1].get("created_at", "")), reverse=True)

        status_counts: Dict[str, int] = {}
        for _, task in entries:
            s = _normalize_task_status(task.get("status"), default="failed")
            status_counts[s] = status_counts.get(s, 0) + 1

        filtered_entries = []
        for tid, task in entries:
            status = _normalize_task_status(task.get("status"), default="failed")
            if status_filter and status != status_filter:
                continue
            if keyword:
                hay = "\n".join(
                    [
                        str(tid),
                        str(task.get("message", "")),
                        str(task.get("current_stage", "")),
                        str(task.get("status", "")),
                        str(task.get("error", "")),
                    ]
                ).lower()
                if keyword not in hay:
                    continue
            filtered_entries.append((tid, task))

        total_filtered = len(filtered_entries)
        sliced = filtered_entries[offset : offset + limit]

        items: list[Dict[str, Any]] = []
        for tid, task in sliced:
            item = {
                "task_id": tid,
                "status": _normalize_task_status(task.get("status"), default="failed"),
                "progress": _clamp_progress(task.get("progress", 0.0)),
                "current_stage": _safe_message(task.get("current_stage", "等待执行"), "等待执行"),
                "message": _safe_message(task.get("message", ""), ""),
                "created_at": task.get("created_at"),
                "updated_at": task.get("updated_at"),
                "output_dir": str(task.get("output_dir", "")),
                "failure": _classify_failure(task),
                "can_retry": bool(
                    str(task.get("video_path", "")).strip()
                    and (PROJECT_ROOT / str(task.get("video_path", ""))).exists()
                ),
            }
            items.append(item)

        return jsonify(
            {
                "ok": True,
                "count": len(items),
                "total": total_filtered,
                "offset": offset,
                "limit": limit,
                "has_more": (offset + len(items)) < total_filtered,
                "status_counts": status_counts,
                "query": keyword,
                "tasks": items,
            }
        )

    @app.get("/api/diagnostic/<task_id>")
    def diagnostic(task_id: str):
        task = _task_snapshot(task_id)
        if not task:
            return jsonify({"ok": False, "error": "task not found", "task_id": task_id}), 404

        failure = _classify_failure(task)
        out_dir = PROJECT_ROOT / str(task.get("output_dir", ""))
        outputs = task.get("outputs", {}) if isinstance(task.get("outputs"), dict) else {}
        existing_outputs: Dict[str, str] = {}
        for key, rel in outputs.items():
            rel_s = str(rel)
            abs_p = PROJECT_ROOT / rel_s
            if abs_p.exists() and abs_p.is_file():
                existing_outputs[key] = rel_s

        traceback_text = str(task.get("error", "") or "")
        traceback_tail = traceback_text[-4000:] if traceback_text else ""

        payload = {
            "ok": True,
            "task_id": task_id,
            "status": _normalize_task_status(task.get("status"), default="failed"),
            "current_stage": _safe_message(task.get("current_stage", "等待执行"), "等待执行"),
            "message": _safe_message(task.get("message", ""), ""),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
            "video_path": str(task.get("video_path", "")),
            "output_dir": str(task.get("output_dir", "")),
            "failure": failure,
            "options": task.get("options", {}),
            "existing_outputs": existing_outputs,
            "output_dir_exists": bool(out_dir.exists()),
            "traceback_tail": traceback_tail,
        }

        if request.args.get("download", "0").strip() in {"1", "true", "yes"}:
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            bio = BytesIO(raw)
            filename = f"{task_id}_diagnostic.json"
            return send_file(bio, as_attachment=True, download_name=filename, mimetype="application/json")
        return jsonify(payload)

    @app.get("/api/progress")
    def progress_sse():
        watch_task = request.args.get("task_id", "").strip()
        start_from = request.args.get("start", "").strip().lower()

        def event_stream():
            idx = max(len(PROGRESS_EVENTS) - 80, 0) if start_from == "recent" else 0
            while True:
                with PROGRESS_COND:
                    if idx >= len(PROGRESS_EVENTS):
                        PROGRESS_COND.wait(timeout=10.0)
                    if idx < len(PROGRESS_EVENTS):
                        event = PROGRESS_EVENTS[idx]
                        idx += 1
                    else:
                        heartbeat_status = "running"
                        heartbeat_progress = None
                        if watch_task:
                            snap = _task_snapshot(watch_task)
                            if snap:
                                heartbeat_status = _normalize_task_status(snap.get("status"), default="running")
                                heartbeat_progress = _clamp_progress(snap.get("progress", 0.0))
                        event = {
                            "type": "heartbeat",
                            "task_id": watch_task or "",
                            "status": heartbeat_status,
                            "progress": heartbeat_progress,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                        }

                if watch_task and event.get("task_id") != watch_task and event.get("type") != "heartbeat":
                    continue

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                time.sleep(0.02)

        return Response(event_stream(), mimetype="text/event-stream")

    @app.get("/api/artifact/<task_id>")
    def artifact(task_id: str):
        task = _task_snapshot(task_id)
        if not task:
            return jsonify({"ok": False, "error": "task not found", "task_id": task_id}), 404

        rel = request.args.get("path", "").replace("\\", "/").strip().lstrip("/")
        if not rel:
            return jsonify({"ok": False, "error": "missing query param path"}), 400

        out_dir = _resolve_project_file(str(task.get("output_dir", "")))
        if out_dir is None or not out_dir.exists():
            return jsonify({"ok": False, "error": "output directory not found"}), 404
        candidate = (out_dir / rel).resolve()
        out_resolved = out_dir.resolve()
        try:
            candidate.relative_to(out_resolved)
        except Exception:
            return jsonify({"ok": False, "error": "illegal path"}), 400
        if not candidate.exists() or not candidate.is_file():
            return jsonify({"ok": False, "error": "artifact not found"}), 404

        mime = "video/mp4" if candidate.suffix.lower() == ".mp4" else None
        return send_file(candidate, as_attachment=False, mimetype=mime)

    @app.get("/api/download/<task_id>/<file_type>")
    def download(task_id: str, file_type: str):
        task = _task_snapshot(task_id)
        if not task:
            return jsonify({"ok": False, "error": "task not found", "task_id": task_id}), 404
        options = task.get("options", {}) if isinstance(task.get("options"), dict) else {}
        output_dir_rel = str(task.get("output_dir", "")).strip()
        out_dir = _resolve_project_file(output_dir_rel) if output_dir_rel else None
        outputs = task.get("outputs", {}) if isinstance(task.get("outputs"), dict) else {}

        aliases = {
            "keyframes_json": "keyframe_json",
            "summary_txt": "summary",
        }
        requested_type = str(file_type or "").strip().lower()
        canonical_type = aliases.get(requested_type, requested_type)

        supported_types = {
            "annotated_video",
            "summary",
            "pdf",
            "alarm_log",
            "keyframe_json",
            "analysis_json",
            "task_result",
        }
        if canonical_type not in supported_types:
            return jsonify(
                {
                    "ok": False,
                    "task_id": task_id,
                    "file_type": canonical_type,
                    "error": f"unsupported file_type: {file_type}",
                    "supported_file_types": sorted(supported_types),
                }
            ), 400

        if canonical_type == "annotated_video" and not bool(options.get("enable_video_export", True)):
            return jsonify(
                {
                    "ok": False,
                    "task_id": task_id,
                    "file_type": canonical_type,
                    "error": "annotated video export was disabled for this task",
                }
            ), 400

        if out_dir is None:
            return jsonify(
                {
                    "ok": False,
                    "task_id": task_id,
                    "file_type": canonical_type,
                    "error": "task output directory is missing",
                }
            ), 404

        mapping = {
            "annotated_video": str(outputs.get("annotated_video") or _safe_rel(out_dir / "hand_annotated.mp4")),
            "pdf": str(outputs.get("report") or _safe_rel(out_dir / "integrated_analysis_report.pdf")),
            "keyframe_json": str(outputs.get("part1_keyframes") or _safe_rel(out_dir / "part1_keyframes.json")),
            "summary": str(outputs.get("overall_summary") or _safe_rel(out_dir / "overall_summary.txt")),
            "alarm_log": str(outputs.get("alarm_log") or _safe_rel(out_dir / "alarm_log.json")),
            "analysis_json": str(outputs.get("analysis_json") or _safe_rel(out_dir / "keyframe_ai_analysis.json")),
            "task_result": str(outputs.get("task_result") or _safe_rel(out_dir / "task_result.json")),
        }
        rel = mapping.get(canonical_type, "")
        path = _resolve_project_file(rel)

        if path is None or not path.exists() or not path.is_file():
            return jsonify(
                {
                    "ok": False,
                    "task_id": task_id,
                    "file_type": canonical_type,
                    "error": f"file not ready: {canonical_type}",
                }
            ), 404
        return send_file(path, as_attachment=True, download_name=path.name)

    @app.get("/api/download_bundle/<task_id>")
    def download_bundle(task_id: str):
        task = _task_snapshot(task_id)
        if not task:
            return jsonify({"ok": False, "error": "task not found", "task_id": task_id}), 404

        out_dir = _resolve_project_file(str(task.get("output_dir", "")))
        outputs = task.get("outputs", {}) if isinstance(task.get("outputs"), dict) else {}
        if out_dir is None or not out_dir.exists():
            return jsonify({"ok": False, "error": "output directory not found"}), 404

        candidates: Dict[str, str] = {
            "annotated_video": outputs.get("annotated_video", ""),
            "report": outputs.get("report", ""),
            "part1_keyframes": outputs.get("part1_keyframes", ""),
            "analysis_json": outputs.get("analysis_json", ""),
            "overall_summary": outputs.get("overall_summary", ""),
            "alarm_log": outputs.get("alarm_log", ""),
            "task_result": outputs.get("task_result", ""),
        }

        existing_files: list[Path] = []
        for rel in candidates.values():
            if not rel:
                continue
            p = _resolve_project_file(str(rel))
            if p and p.exists() and p.is_file():
                existing_files.append(p)

        if not existing_files:
            return jsonify({"ok": False, "error": "no downloadable outputs ready"}), 404

        bundle_path = out_dir / f"{task_id}_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in existing_files:
                arcname = _safe_rel(p)
                zf.write(p, arcname=arcname)

        return send_file(bundle_path, as_attachment=True, download_name=f"{task_id}_bundle.zip")

    @app.get("/api/health")
    def health():
        modules = _probe_modules()
        api_ready = _is_openai_key_configured()
        ffmpeg_available = bool(_resolve_ffmpeg_executable())
        degraded_reasons = []
        if not modules.get("cv2", False):
            degraded_reasons.append("opencv_missing")
        if not modules.get("mediapipe", False):
            degraded_reasons.append("mediapipe_missing(hand detection will be skipped)")
        if not modules.get("reportlab", False):
            degraded_reasons.append("reportlab_missing(pdf will fallback to txt)")
        if not api_ready:
            degraded_reasons.append("missing_DOUBAO_API_KEY_or_ARK_API_KEY(cloud ai unavailable, local fallback is used)")
        if not ffmpeg_available:
            degraded_reasons.append("ffmpeg_missing(video export may fallback to opencv)")
        return jsonify(
            {
                "status": "ok",
                "service": "integrated_system",
                "port": cfg.port,
                "api_key_configured": api_ready,
                "ffmpeg_available": ffmpeg_available,
                "available_modules": modules,
                "task_count": len(TASKS),
                "degraded_reasons": degraded_reasons,
                "defaults": {
                    "enable_hand_detection": cfg.enable_hand_detection,
                    "enable_ai_analysis": cfg.enable_ai_analysis,
                    "enable_pdf": cfg.enable_pdf,
                    "enable_step_check": cfg.enable_step_check,
                    "enable_video_export": cfg.enable_video_export,
                },
            }
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=SETTINGS.host, port=SETTINGS.port, debug=SETTINGS.debug, threaded=True)

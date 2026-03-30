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
from datetime import datetime
from typing import Any, Dict, Optional

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

SETTINGS = load_settings()
EXECUTOR = ThreadPoolExecutor(max_workers=SETTINGS.max_workers, thread_name_prefix="integrated-worker")


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _emit_progress(task_id: str, progress: float, stage: str, message: str) -> None:
    event = {
        "task_id": task_id,
        "progress": max(0.0, min(100.0, float(progress))),
        "current_stage": stage,
        "message": message,
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
    output_dir = PROJECT_ROOT / output_dir_rel if output_dir_rel else None
    outputs = dict(task.get("outputs", {}) if isinstance(task.get("outputs"), dict) else {})

    keyframe_images: list[str] = []
    if output_dir and output_dir.exists():
        part1 = output_dir / "part1_keyframes.json"
        if part1.exists():
            try:
                part1_data = json.loads(part1.read_text(encoding="utf-8"))
                for item in part1_data.get("keyframes", []):
                    img_name = str(item.get("image", "")).strip()
                    if img_name:
                        keyframe_images.append(f"keyframes/{img_name}")
            except Exception:
                pass
        if not keyframe_images:
            for p in sorted((output_dir / "keyframes").glob("keyframe_*.jpg"))[:24]:
                keyframe_images.append(f"keyframes/{p.name}")

    keyframe_urls = [f"/api/artifact/{task_id}?path={rel}" for rel in keyframe_images]

    annotated_rel = str(outputs.get("annotated_video") or "")
    annotated_abs = (PROJECT_ROOT / annotated_rel) if annotated_rel else None
    annotated_exists = bool(annotated_abs and annotated_abs.exists())
    annotated_artifact_path = ""
    if annotated_exists and output_dir:
        try:
            annotated_artifact_path = str(annotated_abs.relative_to(output_dir)).replace("\\", "/")
        except Exception:
            annotated_artifact_path = "hand_annotated.mp4"
    summary_rel = str(outputs.get("overall_summary") or "")
    summary_text = _safe_read_text(PROJECT_ROOT / summary_rel) if summary_rel else ""
    alarm_rel = str(outputs.get("alarm_log") or "")
    alarm_exists = bool(alarm_rel and (PROJECT_ROOT / alarm_rel).exists())
    report_rel = str(outputs.get("report") or "")
    report_exists = bool(report_rel and (PROJECT_ROOT / report_rel).exists())
    report_is_pdf = bool(report_exists and Path(report_rel).suffix.lower() == ".pdf")

    public = {
        "task_id": task_id,
        "status": str(task.get("status", "unknown")),
        "progress": float(task.get("progress", 0.0) or 0.0),
        "current_stage": str(task.get("current_stage", "unknown")),
        "message": str(task.get("message", "")),
        "outputs": outputs,
        "output_dir": output_dir_rel,
        "video_path": str(task.get("video_path", "")),
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
            "keyframe_count": len(keyframe_urls),
            "summary_text": summary_text,
            "alarm_log_exists": alarm_exists,
            "pdf_exists": report_exists,
            "pdf_generated": report_is_pdf,
            "report_path": report_rel,
        },
    }
    return public


def _classify_failure(task: Dict[str, Any]) -> Optional[Dict[str, str]]:
    status = str(task.get("status", "")).strip().lower()
    if status != "failed":
        return None

    stage = str(task.get("current_stage", "")).strip() or "unknown"
    message = str(task.get("message", "")).strip()
    error = str(task.get("error", "")).strip()
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
    elif "api key" in blob or "auth" in blob or "unauthorized" in blob or "forbidden" in blob:
        category = "auth_or_api_error"
        level = "P1"
        title = "Auth/API Configuration Error"
        hint = "Check API key variables and endpoint configuration before retry."
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
    if stage_norm == "hand_detection" and category == "runtime_error":
        title = "Hand Detection Failure"
        hint = "Check mediapipe/opencv runtime and video codec compatibility."
    elif stage_norm == "keyframe_extract" and category == "runtime_error":
        title = "Keyframe/AI Analysis Failure"
        hint = "Check keyframe extraction parameters and AI model endpoint availability."
    elif stage_norm == "step_check" and category == "runtime_error":
        title = "SOP Step Check Failure"
        hint = "Validate SOP rules YAML and keyframe analysis payload format."
    elif stage_norm == "pdf" and category == "runtime_error":
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
    try:
        _update_task(task_id, status="running", progress=3, current_stage="init", message="Task started")
        _emit_progress(task_id, 3, "init", "Task started")

        outputs: Dict[str, str] = {}
        hand_payload: Dict[str, Any] = {}
        keyframe_payload: Dict[str, Any] = {}
        alarm_payload: Dict[str, Any] = {}

        if options.get("enable_hand_detection", True):
            _update_task(task_id, progress=12, current_stage="hand_detection", message="Running hand detection")
            _emit_progress(task_id, 12, "hand_detection", "Running MediaPipe hand detection")
            try:
                hand_payload = run_hand_detection(
                    video_path=str(video_path),
                    output_dir=output_dir,
                    enable_video_export=bool(options.get("enable_video_export", True)),
                )
                outputs["hand_json"] = _safe_rel(output_dir / "hand_detection.json")
                if hand_payload.get("annotated_video"):
                    outputs["annotated_video"] = _safe_rel(Path(str(hand_payload["annotated_video"])))
                _update_task_outputs(task_id, outputs)
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
                _emit_progress(task_id, 14, "hand_detection", f"Hand detection skipped: {hand_exc}")
        else:
            _emit_progress(task_id, 12, "hand_detection", "Hand detection skipped")

        _update_task(task_id, progress=34, current_stage="keyframe_extract", message="Extracting keyframes")
        _emit_progress(task_id, 34, "keyframe_extract", "Extracting keyframes")
        keyframe_payload = run_keyframe_ai_pipeline(
            video_path=str(video_path),
            output_dir=output_dir,
            diff_threshold=settings.keyframe_diff_threshold,
            min_interval_sec=settings.keyframe_min_interval_sec,
            max_keyframes=settings.max_keyframes,
            enable_ai_analysis=bool(options.get("enable_ai_analysis", True)),
            ai_model=settings.ai_model,
            ai_base_url=settings.ai_base_url,
        )
        outputs["part1_keyframes"] = _safe_rel(output_dir / "part1_keyframes.json")
        outputs["overall_summary"] = _safe_rel(output_dir / "overall_summary.txt")
        outputs["analysis_json"] = _safe_rel(output_dir / "keyframe_ai_analysis.json")
        _update_task_outputs(task_id, outputs)

        if options.get("enable_step_check", True):
            _update_task(task_id, progress=67, current_stage="step_check", message="Checking SOP step order")
            _emit_progress(task_id, 67, "step_check", "Checking SOP step order")
            rules_cfg = _load_rules_config()
            expected = _read_rules_expected_steps()
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
        else:
            _emit_progress(task_id, 67, "step_check", "Step check skipped")

        if options.get("enable_pdf", True):
            _update_task(task_id, progress=84, current_stage="pdf", message="Generating PDF report")
            _emit_progress(task_id, 84, "pdf", "Generating PDF report")
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
        else:
            _emit_progress(task_id, 84, "pdf", "PDF generation skipped")

        final_payload = {
            "task_id": task_id,
            "status": "completed",
            "progress": 100.0,
            "current_stage": "completed",
            "message": "Analysis completed",
            "outputs": outputs,
            "output_dir": _safe_rel(output_dir),
            "video_path": _safe_rel(video_path),
            "hand_summary": hand_payload.get("summary", {}),
            "alarm_count": len(alarm_payload.get("alarms", [])),
        }
        (output_dir / "task_result.json").write_text(
            json.dumps(final_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs["task_result"] = _safe_rel(output_dir / "task_result.json")

        _update_task(task_id, **{k: v for k, v in final_payload.items() if k != "task_id"})
        _emit_progress(task_id, 100.0, "completed", "Analysis completed")
    except Exception as exc:
        _update_task(
            task_id,
            status="failed",
            progress=100.0,
            current_stage="failed",
            message=str(exc),
            error=traceback.format_exc(),
            outputs=outputs if "outputs" in locals() else {},
        )
        _emit_progress(task_id, 100.0, "failed", f"Task failed: {exc}")


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
            "current_stage": "queued",
            "message": "Task queued",
            "outputs": {},
            "output_dir": _safe_rel(out_dir),
            "video_path": _safe_rel(input_video),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "options": options,
        }
        with TASKS_LOCK:
            TASKS[task_id] = task

        EXECUTOR.submit(_run_pipeline, task_id, input_video, out_dir, options, cfg)
        _emit_progress(task_id, 0, "queued", "Task queued")

        return jsonify(
            {
                "ok": True,
                "task_id": task_id,
                "status": "pending",
                "progress": 0.0,
                "current_stage": "queued",
                "message": "Task accepted",
                "outputs": {},
                "output_dir": _safe_rel(out_dir),
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

        task = {
            "task_id": new_task_id,
            "status": "pending",
            "progress": 0.0,
            "current_stage": "queued",
            "message": f"Retry queued from {task_id}",
            "outputs": {},
            "output_dir": _safe_rel(out_dir),
            "video_path": _safe_rel(input_video),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "options": options,
            "retry_of": task_id,
        }
        with TASKS_LOCK:
            TASKS[new_task_id] = task

        EXECUTOR.submit(_run_pipeline, new_task_id, input_video, out_dir, options, cfg)
        _emit_progress(new_task_id, 0, "queued", f"Retry queued from {task_id}")

        return jsonify(
            {
                "ok": True,
                "task_id": new_task_id,
                "status": "pending",
                "progress": 0.0,
                "current_stage": "queued",
                "message": f"Retry accepted from {task_id}",
                "outputs": {},
                "output_dir": _safe_rel(out_dir),
                "source_task_id": task_id,
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
                    "status": "not_found",
                    "progress": 0.0,
                    "current_stage": "not_found",
                    "message": "task not found",
                    "outputs": {},
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
            s = str(task.get("status", "unknown")).lower()
            status_counts[s] = status_counts.get(s, 0) + 1

        filtered_entries = []
        for tid, task in entries:
            status = str(task.get("status", "unknown")).lower()
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
                "status": str(task.get("status", "unknown")),
                "progress": float(task.get("progress", 0.0) or 0.0),
                "current_stage": str(task.get("current_stage", "unknown")),
                "message": str(task.get("message", "")),
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
            "status": str(task.get("status", "unknown")),
            "current_stage": str(task.get("current_stage", "")),
            "message": str(task.get("message", "")),
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
                        event = {"type": "heartbeat", "timestamp": datetime.utcnow().isoformat() + "Z"}

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

        out_dir = PROJECT_ROOT / str(task.get("output_dir", ""))
        candidate = (out_dir / rel).resolve()
        out_resolved = out_dir.resolve()
        try:
            candidate.relative_to(out_resolved)
        except Exception:
            return jsonify({"ok": False, "error": "illegal path"}), 400
        if not candidate.exists() or not candidate.is_file():
            return jsonify({"ok": False, "error": "artifact not found"}), 404

        return send_file(candidate, as_attachment=False)

    @app.get("/api/download/<task_id>/<file_type>")
    def download(task_id: str, file_type: str):
        task = _task_snapshot(task_id)
        if not task:
            return jsonify({"ok": False, "error": "task not found", "task_id": task_id}), 404
        options = task.get("options", {}) if isinstance(task.get("options"), dict) else {}
        if file_type == "annotated_video" and not bool(options.get("enable_video_export", True)):
            return jsonify({"ok": False, "error": "annotated video export was disabled for this task"}), 400

        out_dir = PROJECT_ROOT / task.get("output_dir", "")
        outputs = task.get("outputs", {}) if isinstance(task.get("outputs"), dict) else {}

        mapping = {
            "annotated_video": outputs.get("annotated_video") or _safe_rel(out_dir / "hand_annotated.mp4"),
            "pdf": outputs.get("report") or _safe_rel(out_dir / "integrated_analysis_report.pdf"),
            "keyframes_json": outputs.get("part1_keyframes") or _safe_rel(out_dir / "part1_keyframes.json"),
            "summary_txt": outputs.get("overall_summary") or _safe_rel(out_dir / "overall_summary.txt"),
            "alarm_log": outputs.get("alarm_log") or _safe_rel(out_dir / "alarm_log.json"),
            "analysis_json": outputs.get("analysis_json") or _safe_rel(out_dir / "keyframe_ai_analysis.json"),
            "task_result": outputs.get("task_result") or _safe_rel(out_dir / "task_result.json"),
        }
        rel = mapping.get(file_type)
        if not rel:
            return jsonify({"ok": False, "error": f"unsupported file_type: {file_type}"}), 400

        path = PROJECT_ROOT / rel
        if not path.exists():
            return jsonify({"ok": False, "error": f"file not ready: {file_type}"}), 404
        return send_file(path, as_attachment=True)

    @app.get("/api/download_bundle/<task_id>")
    def download_bundle(task_id: str):
        task = _task_snapshot(task_id)
        if not task:
            return jsonify({"ok": False, "error": "task not found", "task_id": task_id}), 404

        out_dir = PROJECT_ROOT / task.get("output_dir", "")
        outputs = task.get("outputs", {}) if isinstance(task.get("outputs"), dict) else {}
        if not out_dir.exists():
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
            p = PROJECT_ROOT / rel
            if p.exists() and p.is_file():
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
        degraded_reasons = []
        if not modules.get("cv2", False):
            degraded_reasons.append("opencv_missing")
        if not modules.get("mediapipe", False):
            degraded_reasons.append("mediapipe_missing(hand detection will be skipped)")
        if not modules.get("reportlab", False):
            degraded_reasons.append("reportlab_missing(pdf will fallback to txt)")
        if not api_ready:
            degraded_reasons.append("missing_DOUBAO_API_KEY_or_ARK_API_KEY(ai analysis will be skipped)")
        return jsonify(
            {
                "status": "ok",
                "service": "integrated_system",
                "port": cfg.port,
                "api_key_configured": api_ready,
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

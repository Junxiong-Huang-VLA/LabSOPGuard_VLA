from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
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
            hand_payload = run_hand_detection(
                video_path=str(video_path),
                output_dir=output_dir,
                enable_video_export=bool(options.get("enable_video_export", True)),
            )
            outputs["hand_json"] = _safe_rel(output_dir / "hand_detection.json")
            if hand_payload.get("annotated_video"):
                outputs["annotated_video"] = _safe_rel(Path(str(hand_payload["annotated_video"])))
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
                "output_dir": _safe_rel(out_dir),
            }
        )

    @app.get("/api/status/<task_id>")
    def status(task_id: str):
        task = _task_snapshot(task_id)
        if not task:
            return jsonify({"ok": False, "error": "task not found", "task_id": task_id}), 404
        return jsonify({"ok": True, **task})

    @app.get("/api/progress")
    def progress_sse():
        watch_task = request.args.get("task_id", "").strip()

        def event_stream():
            idx = 0
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

    @app.get("/api/download/<task_id>/<file_type>")
    def download(task_id: str, file_type: str):
        task = _task_snapshot(task_id)
        if not task:
            return jsonify({"ok": False, "error": "task not found", "task_id": task_id}), 404

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

    @app.get("/api/health")
    def health():
        api_ready = bool(os.getenv("DOUBAO_API_KEY") or os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY"))
        return jsonify(
            {
                "status": "ok",
                "service": "integrated_system",
                "port": cfg.port,
                "api_key_configured": api_ready,
                "task_count": len(TASKS),
            }
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=SETTINGS.host, port=SETTINGS.port, debug=SETTINGS.debug, threaded=True)


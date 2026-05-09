from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel


logger = logging.getLogger(__name__)

AuthDependency = Callable[..., Dict[str, Any]]
PathGetter = Callable[[], Path]
ObjectGetter = Callable[[], Any]


class AttachVideoAnalysisRequest(BaseModel):
    run_experiment_outputs: bool = True
    qwen_frame_writeback_enabled: Optional[bool] = None
    qwen_frame_writeback_limit: Optional[int] = None
    qwen_frame_writeback_force_live: Optional[bool] = None


def _ensure_dir(get_path: PathGetter) -> Path:
    path = get_path()
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_video_analysis_router(
    *,
    require_operator_context: AuthDependency,
    get_video_analysis_pipeline: ObjectGetter,
    get_video_task_store: ObjectGetter,
    get_runtime_settings: ObjectGetter,
    get_output_dir: PathGetter,
    get_upload_dir: PathGetter,
    max_video_upload_bytes: int,
    sanitize_upload_filename: Callable[..., str],
    save_upload_file: Callable[..., Any],
    enforce_experiment_scope: Callable[[Dict[str, Any], str], str],
    get_experiment_dict: Callable[[str], Any],
    resolve_model_path: Callable[[Optional[str]], Optional[str]],
    qwen_writeback_config_from_request: Callable[[Any], Any],
    attach_artifacts_to_experiment: Callable[..., Dict[str, Any]],
    classify_processing_error: Callable[[Exception], Dict[str, str]],
    serve_project_file: Callable[..., Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/v1/video-analysis/analyze", tags=["video-analysis"])
    async def analyze_video_with_visualization(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        yolo_model: Optional[str] = Form(None),
        sample_interval: float = Form(3.0),
        max_frames: int = Form(10),
        experiment_id: Optional[str] = Form(None),
        qwen_frame_writeback_enabled: Optional[bool] = Form(None),
        qwen_frame_writeback_limit: Optional[int] = Form(None),
        qwen_frame_writeback_force_live: Optional[bool] = Form(None),
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        pipeline_cls = get_video_analysis_pipeline()
        if pipeline_cls is None:
            raise HTTPException(status_code=503, detail="Video analysis pipeline is not available")
        task_store = get_video_task_store()
        if task_store is None:
            raise HTTPException(status_code=503, detail="Video analysis task store is not available")

        task_id = str(uuid.uuid4())
        safe_experiment_id = enforce_experiment_scope(auth_ctx, experiment_id) if experiment_id else None
        if safe_experiment_id:
            await get_experiment_dict(safe_experiment_id)

        upload_dir = _ensure_dir(get_upload_dir)
        output_dir = _ensure_dir(get_output_dir)
        safe_video_name = sanitize_upload_filename(video.filename, default_name="video_upload.mp4")
        video_path = upload_dir / f"{task_id}_{safe_video_name}"
        await save_upload_file(video, video_path, max_bytes=max_video_upload_bytes)

        output_video = output_dir / f"annotated_{task_id}.mp4"
        output_json = output_dir / f"analysis_{task_id}.json"

        task_store.create(
            task_id,
            {
                "status": "queued",
                "progress": 0.05,
                "current_stage": "uploaded",
                "video_filename": safe_video_name,
                "video_path": str(video_path),
                "experiment_id": safe_experiment_id,
                "sample_interval": sample_interval,
                "max_frames": max_frames,
                "output_paths": {},
            },
        )

        def run_analysis() -> None:
            try:
                settings = get_runtime_settings()
                yolo_path = resolve_model_path(yolo_model)
                logger.info("Using standalone video-analysis YOLO model: %s", yolo_path)
                task_store.update(task_id, status="running", progress=0.15, current_stage="pipeline_initialized")

                pipeline = pipeline_cls(
                    settings=settings,
                    yolo_model_path=yolo_path,
                    vlm_api_key=os.environ.get("DASHSCOPE_API_KEY"),
                    vlm_base_url=os.environ.get("DASHSCOPE_BASE_URL"),
                    sample_interval=sample_interval,
                    max_frames=max_frames,
                )
                analyses = pipeline.analyze_video(str(video_path))
                task_store.update(task_id, progress=0.7, current_stage="analysis_completed")

                output_json.write_text(
                    json.dumps(pipeline.export_json(analyses), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                task_store.update(
                    task_id,
                    progress=0.82,
                    current_stage="rendering_annotated_video",
                    json_ready=True,
                    output_json=str(output_json),
                )

                pipeline.create_annotated_video(str(video_path), analyses, str(output_video))
                task_store.update(
                    task_id,
                    progress=0.95,
                    current_stage="finalizing",
                    video_ready=output_video.exists(),
                    output_video=str(output_video),
                )

                experiment_output_paths = None
                if safe_experiment_id:
                    task_store.update(task_id, progress=0.97, current_stage="writing_back_to_experiment")
                    qwen_config = qwen_writeback_config_from_request(None)
                    if qwen_frame_writeback_enabled is not None:
                        qwen_config.enabled = bool(qwen_frame_writeback_enabled)
                    if qwen_frame_writeback_limit is not None:
                        qwen_config.limit = max(0, int(qwen_frame_writeback_limit))
                    if qwen_frame_writeback_force_live is not None:
                        qwen_config.force_live = bool(qwen_frame_writeback_force_live)
                    experiment_output_paths = attach_artifacts_to_experiment(
                        experiment_id=safe_experiment_id,
                        task_id=task_id,
                        source_video=video_path,
                        analysis_json=output_json,
                        annotated_video=output_video,
                        sample_interval=sample_interval,
                        max_frames=max_frames,
                        run_experiment_outputs=True,
                        qwen_writeback_config=qwen_config,
                    )

                task_store.update(
                    task_id,
                    status="completed",
                    current_stage="completed",
                    progress=1.0,
                    video_ready=output_video.exists(),
                    json_ready=output_json.exists(),
                    output_video=str(output_video),
                    output_json=str(output_json),
                    output_paths={
                        "analysis_json": str(output_json),
                        "annotated_video": str(output_video),
                        "source_video": str(video_path),
                    },
                    experiment_id=safe_experiment_id,
                    experiment_output_paths=experiment_output_paths,
                )
            except Exception as exc:
                task_store.update(
                    task_id,
                    status="failed",
                    current_stage="failed",
                    error_message=str(exc),
                    error_type=classify_processing_error(exc)["error_type"],
                )
                logger.exception("Standalone video analysis failed for task %s", task_id)

        background_tasks.add_task(run_analysis)
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Video analysis started in background",
        }

    @router.get("/api/v1/video-analysis/status/{task_id}", tags=["video-analysis"])
    async def get_video_analysis_status(
        task_id: str,
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        _ = auth_ctx
        task_store = get_video_task_store()
        if task_store is None:
            raise HTTPException(status_code=503, detail="Task store not available")
        state = task_store.get(task_id)
        if not state:
            raise HTTPException(status_code=404, detail="Task not found")

        output_dir = _ensure_dir(get_output_dir)
        output_video = output_dir / f"annotated_{task_id}.mp4"
        output_json = output_dir / f"analysis_{task_id}.json"
        state["video_ready"] = output_video.exists()
        state["json_ready"] = output_json.exists()
        state["video_size"] = output_video.stat().st_size if output_video.exists() else 0
        state["json_size"] = output_json.stat().st_size if output_json.exists() else 0
        if (
            state.get("status") in {"queued", "running"}
            and state["video_ready"]
            and state["json_ready"]
            and state["video_size"] > 0
            and state["json_size"] > 0
            and not (state.get("experiment_id") and not state.get("experiment_output_paths"))
        ):
            completed_state = {
                "status": "completed",
                "current_stage": "completed",
                "progress": 1.0,
                "video_ready": True,
                "json_ready": True,
                "output_video": str(output_video),
                "output_json": str(output_json),
                "output_paths": {
                    "analysis_json": str(output_json),
                    "annotated_video": str(output_video),
                    "source_video": state.get("video_path"),
                },
            }
            task_store.update(task_id, **completed_state)
            state.update(completed_state)
        return state

    @router.post("/api/v1/experiments/{experiment_id}/video-analysis/attach/{task_id}", tags=["experiments"])
    async def attach_video_analysis_to_experiment(
        experiment_id: str,
        task_id: str,
        req: AttachVideoAnalysisRequest,
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        safe_experiment_id = enforce_experiment_scope(auth_ctx, experiment_id)
        await get_experiment_dict(safe_experiment_id)
        task_store = get_video_task_store()
        if task_store is None:
            raise HTTPException(status_code=503, detail="Task store not available")
        state = task_store.get(task_id)
        if not state:
            raise HTTPException(status_code=404, detail="Video analysis task not found")

        output_dir = _ensure_dir(get_output_dir)
        output_video = Path(state.get("output_video") or output_dir / f"annotated_{task_id}.mp4")
        output_json = Path(state.get("output_json") or output_dir / f"analysis_{task_id}.json")
        source_video = Path(state.get("video_path") or "")
        if not output_video.exists() or not output_json.exists() or not source_video.exists():
            raise HTTPException(status_code=409, detail="Video analysis artifacts are not ready")

        qwen_config = qwen_writeback_config_from_request(None)
        if req.qwen_frame_writeback_enabled is not None:
            qwen_config.enabled = bool(req.qwen_frame_writeback_enabled)
        if req.qwen_frame_writeback_limit is not None:
            qwen_config.limit = max(0, int(req.qwen_frame_writeback_limit))
        if req.qwen_frame_writeback_force_live is not None:
            qwen_config.force_live = bool(req.qwen_frame_writeback_force_live)

        try:
            output_paths = attach_artifacts_to_experiment(
                experiment_id=safe_experiment_id,
                task_id=task_id,
                source_video=source_video,
                analysis_json=output_json,
                annotated_video=output_video,
                sample_interval=float(state.get("sample_interval") or 3.0),
                max_frames=int(state.get("max_frames") or 10),
                run_experiment_outputs=req.run_experiment_outputs,
                qwen_writeback_config=qwen_config,
            )
        except Exception as exc:
            logger.exception("Failed to attach video analysis task %s to experiment %s", task_id, experiment_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        task_store.update(
            task_id,
            experiment_id=safe_experiment_id,
            experiment_output_paths=output_paths,
        )
        return {
            "experiment_id": safe_experiment_id,
            "task_id": task_id,
            "output_paths": output_paths,
            "workspace_url": f"/experiments/{safe_experiment_id}/workspace",
        }

    @router.get("/api/v1/video-analysis/download/{task_id}/video", tags=["video-analysis"])
    async def download_annotated_video(
        task_id: str,
        request: Request,
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        _ = auth_ctx
        output_video = _ensure_dir(get_output_dir) / f"annotated_{task_id}.mp4"
        if not output_video.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        return serve_project_file(output_video, request, media_type="video/mp4")

    @router.get("/api/v1/video-analysis/download/{task_id}/json", tags=["video-analysis"])
    async def download_analysis_json(
        task_id: str,
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        _ = auth_ctx
        output_json = _ensure_dir(get_output_dir) / f"analysis_{task_id}.json"
        if not output_json.exists():
            raise HTTPException(status_code=404, detail="JSON not found")
        return FileResponse(
            path=str(output_json),
            media_type="application/json; charset=utf-8",
            filename=f"analysis_{task_id}.json",
        )

    @router.api_route(
        "/api/v1/video-analysis/{removed_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    async def unknown_standalone_video_analysis_path(removed_path: str):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown standalone video analysis endpoint: {removed_path}",
        )

    return router

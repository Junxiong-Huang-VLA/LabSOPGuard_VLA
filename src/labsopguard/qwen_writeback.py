from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


FLASH_PROMPT = """请观察这张实验视频帧，提取结构化实验信息。只返回 JSON，不要使用 Markdown，不要添加解释。
{
  "objects": ["画面中可确认的实验器材、容器、手部、样品或设备"],
  "actions": ["操作者正在执行的动作"],
  "scene_summary": "一句话描述当前实验场景",
  "risk_flags": ["可见的安全风险或不规范操作，没有则为空数组"],
  "state_changes": ["容器、样品、设备或工位状态变化，没有则为空数组"],
  "confidence": 0.0
}
"""


PLUS_REVIEW_PROMPT = """请对关键帧做更严格的复核，确认实验对象、动作和风险点。只返回 JSON，不要使用 Markdown，不要添加解释。
{
  "review_summary": "关键帧复核结论",
  "confirmed_objects": ["确认存在的关键实验对象"],
  "confirmed_actions": ["确认发生的实验动作"],
  "risk_flags": ["需要关注的安全风险或流程偏差，没有则为空数组"],
  "recommended_review_points": ["建议人工复核的细节，没有则为空数组"],
  "confidence": 0.0
}
"""


@dataclass
class QwenFrameWritebackConfig:
    enabled: bool = False
    flash_model: str = "qwen3.6-flash"
    review_model: str = "qwen3.6-plus"
    limit: Optional[int] = 12
    force_live: bool = False
    use_eval_cache: bool = False
    eval_cache_path: Optional[Path] = None
    timeout_sec: int = 90
    retries: int = 1
    fail_pipeline: bool = False

    @classmethod
    def from_env(cls) -> "QwenFrameWritebackConfig":
        return cls(
            enabled=_env_bool("QWEN_FRAME_WRITEBACK_ENABLED", False),
            flash_model=os.getenv("QWEN_FRAME_WRITEBACK_FLASH_MODEL", "qwen3.6-flash"),
            review_model=os.getenv("QWEN_FRAME_WRITEBACK_REVIEW_MODEL", "qwen3.6-plus"),
            limit=_env_int("QWEN_FRAME_WRITEBACK_LIMIT", 12),
            force_live=_env_bool("QWEN_FRAME_WRITEBACK_FORCE_LIVE", False),
            use_eval_cache=_env_bool("QWEN_FRAME_WRITEBACK_USE_EVAL_CACHE", False),
            timeout_sec=_env_int("QWEN_FRAME_WRITEBACK_TIMEOUT_SEC", 90) or 90,
            retries=_env_int("QWEN_FRAME_WRITEBACK_RETRIES", 1) or 1,
            fail_pipeline=_env_bool("QWEN_FRAME_WRITEBACK_FAIL_PIPELINE", False),
        )


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _extract_json_object(text: str) -> Dict[str, Any]:
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


def _extract_dashscope_text(response: Any) -> str:
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


def _file_uri(path: Path) -> str:
    return f"file://{path.resolve().as_posix()}"


def _dashscope_base_url() -> str:
    return (
        os.getenv("DASHSCOPE_BASE_URL")
        or os.getenv("DASHSCOPE_API_BASE_URL")
        or "https://dashscope.aliyuncs.com/api/v1"
    ).replace("/compatible-mode/v1", "/api/v1")


def _call_frame_model(model: str, image_path: Path, prompt: str, *, timeout_sec: int, retries: int) -> Dict[str, Any]:
    import dashscope  # type: ignore

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")
    dashscope.base_http_api_url = _dashscope_base_url()
    last_error: Optional[BaseException] = None
    for attempt in range(1, max(1, retries + 1) + 1):
        started = time.perf_counter()
        try:
            response = dashscope.MultiModalConversation.call(
                api_key=api_key,
                model=model,
                messages=[{"role": "user", "content": [{"image": _file_uri(image_path)}, {"text": prompt}]}],
                temperature=0.1,
                result_format="message",
                timeout=timeout_sec,
            )
            response_time_ms = int((time.perf_counter() - started) * 1000)
            raw = _extract_dashscope_text(response)
            structured = _extract_json_object(raw)
            if structured:
                return {
                    "structured_result": structured,
                    "raw_response": raw,
                    "response_time_ms": response_time_ms,
                    "source": "live_api",
                    "attempts": attempt,
                }
            last_error = RuntimeError("DashScope response did not contain a structured JSON object")
        except Exception as exc:
            last_error = exc
        if attempt <= retries:
            time.sleep(min(2 * attempt, 6))
    raise RuntimeError(f"{model} failed for {image_path}: {last_error}") from last_error


def _is_pass(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"pass", "true", "1", "ok", "success"}


def _load_eval_cache(path: Optional[Path]) -> Dict[tuple[str, str], Dict[str, Any]]:
    cache: Dict[tuple[str, str], Dict[str, Any]] = {}
    if not path or not path.exists():
        return cache
    for row in _read_json(path, []):
        if not _is_pass(row.get("pass_fail")):
            continue
        model = str(row.get("model_name") or "")
        input_path = str(row.get("input_path") or "")
        structured = row.get("structured_result") or row.get("actual")
        if model and input_path and isinstance(structured, dict):
            cache[(model, str(Path(input_path).resolve()))] = {
                "structured_result": structured,
                "raw_response": row.get("raw_response") or "",
                "response_time_ms": int(row.get("response_time_ms") or 0),
                "source": "eval_cache",
            }
    return cache


def _normalize_flash_payload(payload: Dict[str, Any], *, model: str, source: str, response_time_ms: int) -> Dict[str, Any]:
    return {
        "model": model,
        "source": source,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response_time_ms": response_time_ms,
        "objects": payload.get("objects") or [],
        "actions": payload.get("actions") or [],
        "scene_summary": payload.get("scene_summary") or payload.get("summary") or "",
        "risk_flags": payload.get("risk_flags") or [],
        "state_changes": payload.get("state_changes") or [],
        "confidence": payload.get("confidence"),
    }


def _normalize_plus_payload(payload: Dict[str, Any], *, model: str, source: str, response_time_ms: int) -> Dict[str, Any]:
    return {
        "model": model,
        "source": source,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response_time_ms": response_time_ms,
        "review_summary": payload.get("review_summary") or payload.get("scene_summary") or payload.get("summary") or "",
        "confirmed_objects": payload.get("confirmed_objects") or payload.get("objects") or [],
        "confirmed_actions": payload.get("confirmed_actions") or payload.get("actions") or [],
        "risk_flags": payload.get("risk_flags") or [],
        "recommended_review_points": payload.get("recommended_review_points") or payload.get("state_changes") or [],
        "confidence": payload.get("confidence"),
    }


def _merge_preprocessing(preprocessing: Dict[str, Any], by_item_id: Dict[str, Dict[str, Any]]) -> int:
    updated = 0
    for key in ("time_anchored_material_stream", "video_index"):
        for item in preprocessing.get(key, []) or []:
            item_id = str(item.get("item_id") or "")
            if item_id and item_id in by_item_id:
                item["analysis"] = by_item_id[item_id].get("analysis") or {}
                updated += 1
    return updated


def writeback_qwen_frame_analysis(exp_dir: str | Path, *, exp_id: Optional[str] = None, config: Optional[QwenFrameWritebackConfig] = None) -> Dict[str, Any]:
    config = config or QwenFrameWritebackConfig.from_env()
    exp_dir = Path(exp_dir)
    exp_id = exp_id or exp_dir.name
    material_path = exp_dir / "material_stream.json"
    preprocessing_path = exp_dir / "preprocessing.json"
    if not material_path.exists():
        raise FileNotFoundError(material_path)
    material_stream: List[Dict[str, Any]] = _read_json(material_path, [])
    preprocessing: Dict[str, Any] = _read_json(preprocessing_path, {})
    cache = {} if config.force_live or not config.use_eval_cache else _load_eval_cache(config.eval_cache_path)

    processed = 0
    flash_written = 0
    plus_written = 0
    failures: List[Dict[str, Any]] = []
    by_item_id: Dict[str, Dict[str, Any]] = {}

    for item in material_stream:
        if config.limit is not None and config.limit > 0 and processed >= config.limit:
            break
        frame_path_value = item.get("frame_bgr_path")
        if not frame_path_value:
            continue
        frame_path = Path(str(frame_path_value))
        if not frame_path.is_absolute():
            frame_path = PROJECT_ROOT / frame_path
        if not frame_path.exists():
            failures.append({"item_id": item.get("item_id"), "stage": "frame_exists", "error": f"missing frame: {frame_path}"})
            continue
        processed += 1
        analysis = dict(item.get("analysis") or {})

        try:
            cache_key = (config.flash_model, str(frame_path.resolve()))
            flash = cache.get(cache_key) or _call_frame_model(
                config.flash_model,
                frame_path,
                FLASH_PROMPT,
                timeout_sec=config.timeout_sec,
                retries=config.retries,
            )
            analysis["qwen3_6_flash_frame"] = _normalize_flash_payload(
                flash["structured_result"],
                model=config.flash_model,
                source=flash.get("source") or "live_api",
                response_time_ms=int(flash.get("response_time_ms") or 0),
            )
            flash_written += 1
        except Exception as exc:
            failures.append({"item_id": item.get("item_id"), "stage": "flash_frame", "error": str(exc), "frame_path": str(frame_path)})

        if item.get("is_key_frame"):
            try:
                cache_key = (config.review_model, str(frame_path.resolve()))
                plus = cache.get(cache_key) or _call_frame_model(
                    config.review_model,
                    frame_path,
                    PLUS_REVIEW_PROMPT,
                    timeout_sec=config.timeout_sec,
                    retries=config.retries,
                )
                analysis["qwen3_6_plus_keyframe_review"] = _normalize_plus_payload(
                    plus["structured_result"],
                    model=config.review_model,
                    source=plus.get("source") or "live_api",
                    response_time_ms=int(plus.get("response_time_ms") or 0),
                )
                plus_written += 1
            except Exception as exc:
                failures.append({"item_id": item.get("item_id"), "stage": "plus_keyframe_review", "error": str(exc), "frame_path": str(frame_path)})

        item["analysis"] = analysis
        if analysis.get("qwen3_6_flash_frame"):
            flash_payload = analysis["qwen3_6_flash_frame"]
            item["scene_description"] = item.get("scene_description") or flash_payload.get("scene_summary")
            item["object_labels"] = list(dict.fromkeys([*(item.get("object_labels") or []), *[str(v) for v in flash_payload.get("objects") or []]]))
            item["detected_activities"] = list(dict.fromkeys([*(item.get("detected_activities") or []), *[str(v) for v in flash_payload.get("actions") or []]]))
        by_item_id[str(item.get("item_id") or "")] = item

    material_path.write_text(json.dumps(material_stream, ensure_ascii=False, indent=2), encoding="utf-8")
    preprocessing_updates = 0
    if preprocessing_path.exists():
        preprocessing_updates = _merge_preprocessing(preprocessing, by_item_id)
        preprocessing_path.write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8")

    from labsopguard.retrieval import MaterialRetrievalIndex

    index_path = exp_dir / "material_index.sqlite"
    index = MaterialRetrievalIndex(index_path)
    try:
        index.reset()
        index.index_payloads(material_stream, preprocessing=preprocessing, experiment_id=exp_id)
        health = index.health_check()
    finally:
        index.close()

    report = {
        "experiment_id": exp_id,
        "enabled": config.enabled,
        "flash_model": config.flash_model,
        "review_model": config.review_model,
        "limit": config.limit,
        "processed_items": processed,
        "flash_written": flash_written,
        "plus_keyframe_reviews_written": plus_written,
        "preprocessing_updates": preprocessing_updates,
        "failures": failures,
        "material_stream_path": str(material_path),
        "preprocessing_path": str(preprocessing_path),
        "material_index_path": str(index_path),
        "material_index_health": health,
    }
    report_path = exp_dir / "qwen_frame_writeback.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report

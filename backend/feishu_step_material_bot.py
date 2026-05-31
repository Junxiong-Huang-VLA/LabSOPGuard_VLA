from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import quote

try:
    from backend.feishu_notifier import FeishuSendResult
except Exception:  # pragma: no cover - supports running backend/main.py directly.
    from feishu_notifier import FeishuSendResult  # type: ignore


DEFAULT_PUBLIC_BASE_URL = "http://127.0.0.1:8000"


@dataclass
class FeishuStepMaterialPushResult:
    text: str
    image_path: Optional[str]
    feishu: FeishuSendResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "image_path": self.image_path,
            "send_result": self.feishu.to_dict(),
        }


def build_step_query_feishu_message(
    *,
    experiment_id: str,
    experiment_title: str,
    query_result: Mapping[str, Any],
    experiment_dir: str | Path,
    public_base_url: Optional[str] = None,
    max_candidates: int = 3,
) -> str:
    """Build a compact Feishu text message for a step-material query result."""
    exp_dir = Path(experiment_dir)
    base_url = _clean_base_url(public_base_url)
    judgement = _as_mapping(query_result.get("judgement"))
    candidates = [item for item in (query_result.get("candidates") or []) if isinstance(item, Mapping)]
    status = str(judgement.get("status") or "insufficient")
    confidence = _safe_float(judgement.get("confidence"))

    lines = [
        "【实验步骤核验】",
        f"实验：{experiment_title or experiment_id}",
        f"步骤：{query_result.get('step_text') or '-'}",
        f"结论：{_judgement_label(status, judgement)}（置信度 {confidence:.2f}）",
    ]
    message_video_time = query_result.get("message_video_time_sec")
    if message_video_time is not None:
        lines.append(f"时间对齐：飞书消息时间 -> 视频 {_format_video_time(message_video_time)}")
    window = _as_mapping(query_result.get("search_window"))
    if window:
        lines.append(
            "检索窗口："
            f"{_format_video_time(window.get('start_sec'))} - {_format_video_time(window.get('end_sec'))}"
        )
    reason = str(judgement.get("reason") or "").strip()
    if reason:
        lines.append(f"依据：{reason}")

    lines.append("")
    lines.append("关键素材：")
    if not candidates:
        lines.append("未检索到足够的时间锚点素材，需要补充该步骤附近的关键帧或片段。")
    for index, candidate in enumerate(candidates[: max(1, max_candidates)], start=1):
        lines.extend(_candidate_lines(index, candidate, experiment_id, exp_dir, base_url))
    return "\n".join(lines)


def select_evidence_image_path(
    *,
    experiment_dir: str | Path,
    query_result: Mapping[str, Any],
) -> Optional[Path]:
    """Pick the first existing preview/keyframe from query candidates."""
    exp_dir = Path(experiment_dir)
    for candidate in query_result.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        values: list[Any] = []
        values.append(candidate.get("preview_path"))
        values.extend(candidate.get("keyframe_paths") or [])
        if _is_image_path(candidate.get("stored_file")):
            values.append(candidate.get("stored_file"))
        if _is_image_path(candidate.get("source_reference_file")):
            values.append(candidate.get("source_reference_file"))
        for value in values:
            path = _resolve_experiment_file(exp_dir, value)
            if path and path.exists() and path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return path
    return None


def send_step_query_result_to_feishu(
    *,
    notifier: Any,
    experiment_dir: str | Path,
    experiment_id: str,
    experiment_title: str,
    query_result: Mapping[str, Any],
    public_base_url: Optional[str] = None,
    include_evidence_image: bool = True,
) -> FeishuStepMaterialPushResult:
    """Send step judgement text and, when available, one evidence image to Feishu."""
    text = build_step_query_feishu_message(
        experiment_id=experiment_id,
        experiment_title=experiment_title,
        query_result=query_result,
        experiment_dir=experiment_dir,
        public_base_url=public_base_url,
    )
    send_result = FeishuSendResult()
    text_message_id = notifier.send_text(text)
    send_result.text_sent = True
    if text_message_id:
        send_result.message_ids.append(str(text_message_id))

    image_path = select_evidence_image_path(experiment_dir=experiment_dir, query_result=query_result)
    if include_evidence_image and image_path is not None:
        image_key = notifier.upload_image(
            image_path.read_bytes(),
            filename=image_path.name,
            content_type=_image_content_type(image_path),
        )
        send_result.image_key = image_key
        image_message_id = notifier.send_image(image_key)
        send_result.image_sent = True
        if image_message_id:
            send_result.message_ids.append(str(image_message_id))

    return FeishuStepMaterialPushResult(
        text=text,
        image_path=str(image_path) if image_path is not None else None,
        feishu=send_result,
    )


def _candidate_lines(
    index: int,
    candidate: Mapping[str, Any],
    experiment_id: str,
    experiment_dir: Path,
    public_base_url: str,
) -> list[str]:
    event_type = candidate.get("event_type") or candidate.get("asset_type") or "material"
    objects = ", ".join(str(item) for item in candidate.get("objects") or [] if item)
    time_range = f"{_format_video_time(candidate.get('start_sec'))} - {_format_video_time(candidate.get('end_sec'))}"
    summary = f"{index}. {event_type} | {time_range}"
    if objects:
        summary += f" | 对象：{objects}"
    lines = [summary]
    clip_url = _experiment_file_url(
        experiment_id=experiment_id,
        experiment_dir=experiment_dir,
        path_value=candidate.get("clip_path") or (_asset_path_value(candidate, video=True)),
        public_base_url=public_base_url,
    )
    preview_url = _experiment_file_url(
        experiment_id=experiment_id,
        experiment_dir=experiment_dir,
        path_value=candidate.get("preview_path") or _asset_path_value(candidate, image=True),
        public_base_url=public_base_url,
    )
    if clip_url:
        lines.append(f"   片段：{clip_url}")
    if preview_url:
        lines.append(f"   关键帧：{preview_url}")
    material_id = candidate.get("material_id")
    if material_id:
        lines.append(f"   素材ID：{material_id}")
    return lines


def _experiment_file_url(
    *,
    experiment_id: str,
    experiment_dir: Path,
    path_value: Any,
    public_base_url: str,
) -> Optional[str]:
    if not path_value:
        return None
    value = str(path_value).strip()
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("/api/"):
        return f"{public_base_url}{value}"

    path = Path(value)
    if path.is_absolute():
        try:
            value = path.resolve().relative_to(experiment_dir.resolve()).as_posix()
        except ValueError:
            return None
    else:
        value = value.replace("\\", "/").lstrip("/")
        direct = experiment_dir / value
        material_relative = experiment_dir / "material_references" / value
        if not direct.exists() and material_relative.exists():
            value = f"material_references/{value}"
    return f"{public_base_url}/api/v1/experiments/{quote(experiment_id, safe='')}/files/{quote(value, safe='/')}"


def _resolve_experiment_file(experiment_dir: Path, path_value: Any) -> Optional[Path]:
    if not path_value:
        return None
    value = str(path_value).strip()
    if not value or value.startswith(("http://", "https://", "/api/")):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    for target in ((experiment_dir / path).resolve(), (experiment_dir / "material_references" / path).resolve()):
        try:
            target.relative_to(experiment_dir.resolve())
        except ValueError:
            continue
        if target.exists():
            return target
    return (experiment_dir / path).resolve()


def _asset_path_value(candidate: Mapping[str, Any], *, image: bool = False, video: bool = False) -> Any:
    for key in ("stored_file", "source_reference_file", "file_path"):
        value = candidate.get(key)
        if image and _is_image_path(value):
            return value
        if video and _is_video_path(value):
            return value
    return None


def _is_image_path(value: Any) -> bool:
    return Path(str(value or "")).suffix.lower() in {".jpg", ".jpeg", ".png"}


def _is_video_path(value: Any) -> bool:
    return Path(str(value or "")).suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}


def _judgement_label(status: str, judgement: Mapping[str, Any]) -> str:
    label = str(judgement.get("label") or "").strip()
    if label:
        return label
    return {
        "correct": "符合要求",
        "incorrect": "不符合要求",
        "insufficient": "证据不足",
    }.get(status, status or "证据不足")


def _format_video_time(value: Any) -> str:
    seconds = _safe_float(value)
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean_base_url(value: Optional[str]) -> str:
    return (value or DEFAULT_PUBLIC_BASE_URL).strip().rstrip("/") or DEFAULT_PUBLIC_BASE_URL


def _image_content_type(path: Path) -> str:
    if path.suffix.lower() == ".png":
        return "image/png"
    return "image/jpeg"

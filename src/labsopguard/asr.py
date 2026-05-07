from __future__ import annotations

import os
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class ASRUnavailableError(RuntimeError):
    pass


@dataclass
class TranscriptSegment:
    text: str
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    speaker: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_context_input(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "kind": "transcript",
            "source_type": "asr",
            "text": self.text,
            "metadata": self.metadata,
        }
        if self.start_time_sec is not None:
            payload["start_time_sec"] = round(float(self.start_time_sec), 3)
            payload["timestamp_sec"] = round(float(self.start_time_sec), 3)
        if self.end_time_sec is not None:
            payload["end_time_sec"] = round(float(self.end_time_sec), 3)
        if self.speaker:
            payload["speaker"] = self.speaker
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


@dataclass
class TranscriptResult:
    text: str
    segments: List[TranscriptSegment]
    provider: str
    model: str
    language: Optional[str] = None


def _read_api_key() -> Optional[str]:
    return (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
    )


def _dashscope_base_url() -> str:
    base_url = (
        os.getenv("ASR_BASE_URL")
        or os.getenv("DASHSCOPE_API_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
        or os.getenv("QWEN_BASE_URL")
        or "https://dashscope.aliyuncs.com/api/v1"
    )
    return base_url.replace("/compatible-mode/v1", "/api/v1")


def _extract_text(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("text") or "").strip()
    return str(getattr(response, "text", "") or "").strip()


def _extract_segments(response: Any) -> List[TranscriptSegment]:
    raw_segments = response.get("segments") if isinstance(response, dict) else getattr(response, "segments", None)
    segments: List[TranscriptSegment] = []
    for item in raw_segments or []:
        if not isinstance(item, dict):
            item = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                text=text,
                start_time_sec=item.get("start"),
                end_time_sec=item.get("end"),
                confidence=item.get("confidence") or item.get("avg_logprob"),
                metadata={k: v for k, v in item.items() if k not in {"text", "start", "end", "confidence"}},
            )
        )
    return segments


def _extract_dashscope_asr_text(response: Any) -> str:
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


def _audio_file_uri(audio_path: Path) -> str:
    return f"file://{audio_path.resolve().as_posix()}"


def _transcribe_with_qwen(
    audio_path: Path,
    *,
    language: Optional[str],
    prompt: Optional[str],
) -> TranscriptResult:
    api_key = _read_api_key()
    if not api_key:
        raise ASRUnavailableError("DASHSCOPE_API_KEY is not configured.")

    try:
        import dashscope  # type: ignore
    except Exception as exc:
        raise ASRUnavailableError("dashscope package is required for Qwen ASR. Install with: pip install dashscope") from exc

    max_local_mb = float(os.getenv("ASR_LOCAL_FILE_MAX_MB", "10"))
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_local_mb:
        raise ASRUnavailableError(
            f"Qwen qwen3-asr-flash local input is limited to about 10MB; got {file_size_mb:.2f}MB. "
            "Use a public file URL or split the audio before upload."
        )

    model = os.getenv("ASR_MODEL", "qwen3-asr-flash")
    dashscope.base_http_api_url = _dashscope_base_url()
    asr_options: Dict[str, Any] = {"enable_itn": os.getenv("ASR_ENABLE_ITN", "false").lower() == "true"}
    if language:
        asr_options["language"] = language
    if prompt:
        asr_options["hotwords"] = prompt

    response = dashscope.MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=[
            {
                "role": "user",
                "content": [{"audio": _audio_file_uri(audio_path)}],
            }
        ],
        result_format="message",
        asr_options=asr_options,
    )
    text = _extract_dashscope_asr_text(response)
    segments = [TranscriptSegment(text=text, start_time_sec=0.0, metadata={"asr_options": asr_options})] if text else []
    return TranscriptResult(
        text=text,
        segments=segments,
        provider="qwen_dashscope",
        model=model,
        language=language,
    )


def transcribe_audio_file(
    audio_path: str | Path,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> TranscriptResult:
    provider = os.getenv("ASR_PROVIDER", "qwen").strip().lower()
    audio_file = Path(audio_path)
    if provider in {"qwen", "dashscope", "aliyun", "bailian"}:
        return _transcribe_with_qwen(audio_file, language=language, prompt=prompt)
    raise ASRUnavailableError(f"Unsupported ASR_PROVIDER: {provider}. Use ASR_PROVIDER=qwen.")


def asr_diagnostics() -> Dict[str, Any]:
    provider = os.getenv("ASR_PROVIDER", "qwen").strip().lower()
    model = os.getenv("ASR_MODEL", "qwen3-asr-flash")
    api_key_configured = bool(_read_api_key())
    dashscope_installed = importlib.util.find_spec("dashscope") is not None
    if provider not in {"qwen", "dashscope", "aliyun", "bailian"}:
        status = f"unsupported_provider:{provider}"
    elif not api_key_configured:
        status = "missing_api_key"
    elif not dashscope_installed:
        status = "dashscope_missing"
    else:
        status = "configured"
    return {
        "qwen_asr_status": status,
        "current_asr_model": model,
        "asr_provider": provider,
        "dashscope_installed": dashscope_installed,
        "api_key_configured": api_key_configured,
    }

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

# 模型缓存
_cached_whisper_model = None

# Auto-load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


def _load_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None
    return OpenAI


def _read_api_key() -> Optional[str]:
    return os.getenv("DOUBAO_API_KEY") or os.getenv("ARK_API_KEY")


def build_openai_client(base_url: Optional[str] = None):
    openai_cls = _load_openai_client()
    if openai_cls is None:
        return None

    api_key = _read_api_key()
    if not api_key:
        return None

    final_base = base_url or os.getenv("OPENAI_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
    try:
        return openai_cls(api_key=api_key, base_url=final_base)
    except Exception:
        return None


def _img_to_data_url(path: str | Path) -> str:
    p = Path(path)
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _local_visual_analysis(path: Path) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return {
            "summary": "本地视觉分析：运行环境缺少 cv2/numpy，仅保留基础帧记录。",
            "risk_level": "unknown",
        }

    frame = cv2.imread(str(path))
    if frame is None:
        return {
            "summary": "本地视觉分析：图像读取失败，无法提取视觉特征。",
            "risk_level": "unknown",
        }

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 60, 140)
    edge_ratio = float(np.count_nonzero(edges) / edges.size) if edges.size > 0 else 0.0

    hints: List[str] = []
    risk_level = "low"

    if brightness < 45:
        hints.append("画面偏暗")
        risk_level = "medium"
    elif brightness > 220:
        hints.append("画面过曝")
        risk_level = "medium"
    else:
        hints.append("光照基本稳定")

    if sharpness < 40:
        hints.append("清晰度偏低")
        risk_level = "medium"
    else:
        hints.append("清晰度正常")

    if edge_ratio > 0.16:
        hints.append("操作区域变化较大")
    elif edge_ratio < 0.03:
        hints.append("画面变化较小")
    else:
        hints.append("画面变化中等")

    summary = (
        f"本地视觉分析：亮度 {brightness:.1f}、清晰度 {sharpness:.1f}、边缘密度 {edge_ratio:.3f}；"
        + "，".join(hints)
        + "。"
    )
    return {"summary": summary, "risk_level": risk_level}


def _build_local_fallback_result(
    paths: List[Path],
    reason: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    error_detail: str = "",
) -> Dict[str, Any]:
    total = len(paths)
    analyses: List[Dict[str, Any]] = []
    risk_counter = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for idx, p in enumerate(paths, start=1):
        if progress_callback:
            progress_callback(idx - 1, total, f"AI 分析中：云端不可用，已切换本地视觉分析（{idx}/{total}）。")
        local = _local_visual_analysis(p)
        risk = str(local.get("risk_level", "unknown") or "unknown").lower()
        if risk not in risk_counter:
            risk = "unknown"
        risk_counter[risk] += 1
        analyses.append(
            {
                "image": p.name,
                "summary": local.get("summary", "本地视觉分析未返回结果。"),
                "risk_level": risk,
                "cloud_error": error_detail,
            }
        )
        if progress_callback:
            progress_callback(idx, total, f"AI 分析中：本地视觉分析已完成 {idx}/{total} 帧。")

    overall = (
        f"已完成 {total} 帧本地视觉分析（云端模型不可用: {reason}）。"
        f" 风险分布：high {risk_counter['high']} / medium {risk_counter['medium']} / "
        f"low {risk_counter['low']} / unknown {risk_counter['unknown']}。"
    )
    if error_detail:
        overall += f" 远程错误：{error_detail}"
    return {
        "enabled": True,
        "reason": f"local_fallback_{reason}",
        "analyses": analyses,
        "overall_summary": overall,
        "fallback_mode": "local_visual",
        "cloud_error": error_detail,
    }


def _extract_response_text(resp: Any) -> str:
    # Compatible with both plain string content and structured content parts.
    if not resp or not getattr(resp, "choices", None):
        return ""
    first = resp.choices[0]
    msg = getattr(first, "message", None)
    if msg is None:
        return ""

    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text = str(part.get("text", "")).strip()
                if text:
                    chunks.append(text)
            else:
                text = str(getattr(part, "text", "")).strip()
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return str(content or "").strip()


def analyze_keyframes_with_openai(
    image_paths: Iterable[str | Path],
    model: str,
    prompt: Optional[str] = None,
    base_url: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    paths = [Path(p) for p in image_paths]
    if not paths:
        if progress_callback:
            progress_callback(0, 0, "未检测到关键帧，AI 分析已跳过。")
        return {"enabled": False, "reason": "no_keyframes", "analyses": [], "overall_summary": ""}

    openai_cls = _load_openai_client()
    if openai_cls is None:
        return _build_local_fallback_result(paths, "openai_package_missing", progress_callback)

    api_key = _read_api_key()
    if not api_key:
        return _build_local_fallback_result(paths, "missing_api_key", progress_callback)

    final_base = base_url or os.getenv("OPENAI_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
    try:
        client = openai_cls(api_key=api_key, base_url=final_base)
    except Exception as exc:
        return _build_local_fallback_result(paths, "client_init_failed", progress_callback, error_detail=repr(exc))

    user_prompt = prompt or (
        "You are analyzing chemistry lab operation frames. "
        "For each image, return a concise safety observation and inferred step."
    )

    analyses: List[Dict[str, Any]] = []
    cloud_failures = 0
    total = len(paths)
    for idx, p in enumerate(paths, start=1):
        if progress_callback:
            progress_callback(idx - 1, total, f"AI 分析中：第 {idx}/{total} 帧。")
        try:
            content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": _img_to_data_url(p)}},
            ]
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a lab SOP compliance assistant."},
                    {"role": "user", "content": content},
                ],
                temperature=0.2,
            )
            text = _extract_response_text(resp)
            analyses.append(
                {
                    "image": p.name,
                    "summary": text.strip() or "No response text from model.",
                    "risk_level": "review",
                }
            )
        except Exception as exc:
            cloud_failures += 1
            local = _local_visual_analysis(p)
            err_text = repr(exc)
            analyses.append(
                {
                    "image": p.name,
                    "summary": f"云端分析失败，已回退本地视觉分析：{local.get('summary', str(exc))}",
                    "risk_level": str(local.get("risk_level", "unknown") or "unknown"),
                    "cloud_error": err_text,
                }
            )
        if progress_callback:
            progress_callback(idx, total, f"AI 分析中：已完成 {idx}/{total} 帧。")

    joined = "\n".join(f"- {item['image']}: {item['summary']}" for item in analyses)
    cloud_error_details = [str(item.get("cloud_error", "")).strip() for item in analyses if str(item.get("cloud_error", "")).strip()]
    if cloud_failures > 0:
        overall_summary = (
            f"AI 分析完成，共 {len(analyses)} 帧；其中 {cloud_failures} 帧云端调用失败，已自动回退本地视觉分析。\n{joined}"
        )
        if cloud_error_details:
            uniq_errors = []
            for err in cloud_error_details:
                if err not in uniq_errors:
                    uniq_errors.append(err)
            overall_summary += "\n远程错误详情：\n- " + "\n- ".join(uniq_errors[:10])
    else:
        overall_summary = f"AI analyzed {len(analyses)} keyframes.\n{joined}"
    return {
        "enabled": True,
        "reason": "ok" if cloud_failures == 0 else "partial_cloud_failure_with_local_fallback",
        "analyses": analyses,
        "overall_summary": overall_summary,
        "cloud_failures": cloud_failures,
        "cloud_error_details": cloud_error_details,
    }


def text_to_speech(
    text: str,
    voice: str = "alloy",
    model: str = "tts-1",
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    将文本转为语音。
    优先使用本地 edge-tts（免费、高质量），失败时尝试 OpenAI 兼容 API。

    Args:
        text: 要转换的文本
        voice: 音色选项
              - edge-tts: "zh-CN-XiaoxiaoNeural" (女), "zh-CN-YunxiNeural" (男) 等
              - OpenAI: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
        model: 保留参数，用于 OpenAI API
        output_path: 保存路径，默认保存为临时文件

    Returns:
        包含 audio_path 或 error 的字典
    """
    import tempfile
    import asyncio

    if not text or not text.strip():
        return {"error": "文本不能为空"}

    # 优先尝试本地 edge-tts
    try:
        return _edge_tts_convert(text, voice, output_path)
    except Exception as edge_error:
        print(f"edge-tts 失败: {edge_error}, 尝试 pyttsx3...")

    # 回退到 pyttsx3 (Windows SAPI，离线工作)
    try:
        return _pyttsx3_convert(text, voice, output_path)
    except Exception as pyttsx3_error:
        print(f"pyttsx3 失败: {pyttsx3_error}, 尝试 OpenAI API...")

    # 最后回退到 OpenAI API
    return _openai_tts_fallback(text, voice, model, output_path)


def _edge_tts_convert(text: str, voice: str, output_path: str | Path | None) -> Dict[str, Any]:
    """使用 edge-tts 本地转换"""
    import asyncio
    import edge_tts as edge_tts_lib
    import tempfile

    # 映射 voice 参数到 edge-tts 音色
    voice_map = {
        "alloy": "zh-CN-XiaoxiaoNeural",
        "echo": "zh-CN-YunxiNeural",
        "fable": "zh-CN-XiaoyiNeural",
        "onyx": "zh-CN-YunyangNeural",
        "nova": "zh-CN-YunxiaNeural",
        "shimmer": "zh-CN-YunjianNeural",
    }
    edge_voice = voice_map.get(voice, voice)
    # 如果用户指定了完整的 zh-CN-xxxNeural，使用用户指定的
    if not voice.startswith("zh-"):
        edge_voice = voice_map.get(voice, "zh-CN-XiaoxiaoNeural")

    async def _convert():
        nonlocal output_path
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        communicate = edge_tts_lib.Communicate(text, edge_voice)
        await communicate.save(output_path)

    asyncio.run(_convert())
    return {"ok": True, "audio_path": str(output_path), "text": text, "method": "edge-tts"}


def _pyttsx3_convert(text: str, voice: str, output_path: str | Path | None) -> Dict[str, Any]:
    """使用 pyttsx3 本地转换 (Windows SAPI5，离线工作)"""
    import pyttsx3
    import tempfile

    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    else:
        output_path = str(output_path)

    engine = pyttsx3.init()
    # 尝试使用中文语音
    try:
        engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_ZH-CN_HUIHUI_11.0')
    except Exception:
        pass  # 使用默认语音
    engine.setProperty('rate', 150)  # 语速
    engine.setProperty('volume', 1.0)  # 音量

    engine.save_to_file(text, output_path)
    engine.runAndWait()
    engine.stop()

    return {"ok": True, "audio_path": str(output_path), "text": text, "method": "pyttsx3"}


def _openai_tts_fallback(text: str, voice: str, model: str, output_path: str | Path | None) -> Dict[str, Any]:
    """使用 OpenAI 兼容 API (DashScope/Ark) 进行 TTS"""
    import tempfile

    client = build_openai_client()
    if not client:
        return {"error": "无法构建 OpenAI 客户端，请检查 API Key 配置 (DOUBAO_API_KEY 或 ARK_API_KEY)"}

    try:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="mp3"
        )
        audio_bytes = response.read()

        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_bytes)
                output_path = f.name
        else:
            output_path = Path(output_path)
            output_path.write_bytes(audio_bytes)

        return {"ok": True, "audio_path": str(output_path), "text": text, "method": "openai-api"}
    except Exception as e:
        return {"error": f"TTS 调用失败: {str(e)}"}


def speech_to_text(
    audio_path: str | Path,
    language: str = "zh",
    model: str = "whisper-1",
) -> Dict[str, Any]:
    """
    将语音转为文本。
    优先使用本地 faster-whisper (免费、离线)，失败时尝试 OpenAI 兼容 API。

    Args:
        audio_path: 音频文件路径
        language: 语言代码，如 "zh" 表示中文
        model: ASR 模型，默认 whisper-1

    Returns:
        包含 text 或 error 的字典
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        return {"error": f"音频文件不存在: {audio_path}"}

    # 优先尝试本地 faster-whisper
    try:
        return _faster_whisper_convert(str(audio_path), language)
    except Exception as whisper_error:
        print(f"faster-whisper 失败: {whisper_error}, 尝试 OpenAI API...")

    # 回退到 OpenAI API
    return _openai_asr_fallback(audio_path, language, model)


def _faster_whisper_convert(audio_path: str, language: str) -> Dict[str, Any]:
    """使用本地 faster-whisper 进行语音识别"""
    import os
    import faster_whisper
    from faster_whisper import WhisperModel

    # 解决 OpenMP 冲突
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # 模型缓存（全局变量）
    global _cached_whisper_model
    if _cached_whisper_model is None:
        # 模型大小选择：tiny/base/small/medium/large-v2/v3
        # 使用 small 作为平衡（首次运行会自动下载）
        _cached_whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

    model = _cached_whisper_model

    segments, info = model.transcribe(
        audio_path,
        language=language if language else None,
        beam_size=5,
        vad_filter=True,  # 启用语音活动检测
    )

    texts = []
    for segment in segments:
        texts.append(segment.text)

    full_text = "".join(texts)
    detected_lang = info.language if hasattr(info, 'language') else language

    return {"ok": True, "text": full_text, "language": detected_lang, "method": "faster-whisper"}


def _openai_asr_fallback(audio_path: Path, language: str, model: str) -> Dict[str, Any]:
    """使用 OpenAI 兼容 API 进行 ASR"""
    client = build_openai_client()
    if not client:
        return {"error": "无法构建 OpenAI 客户端，请检查 API Key 配置 (DOUBAO_API_KEY 或 ARK_API_KEY)"}

    try:
        with open(audio_path, "rb") as audio_file:
            # 尝试调用 ASR API
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=language if language else None,
                response_format="text"
            )

        text = response.text if hasattr(response, "text") else str(response)
        return {"ok": True, "text": text, "language": language, "method": "openai-api"}

    except Exception as e:
        error_msg = str(e)
        return {"error": f"ASR 调用失败: {error_msg}"}


def stream_text_to_speech(
    text: str,
    voice: str = "alloy",
    model: str = "tts-1",
) -> Dict[str, Any]:
    """
    流式语音合成 - 返回音频数据而不是文件路径。
    适用于需要实时播放的场景。

    Returns:
        包含 audio_data (base64) 或 error 的字典
    """
    client = build_openai_client()
    if not client:
        return {"error": "无法构建 OpenAI 客户端，请检查 API Key 配置"}

    if not text or not text.strip():
        return {"error": "文本不能为空"}

    try:
        import base64

        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="mp3"
        )

        audio_bytes = response.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "ok": True,
            "audio_data": audio_base64,
            "format": "mp3",
            "text": text
        }

    except Exception as e:
        return {"error": f"流式 TTS 调用失败: {str(e)}"}

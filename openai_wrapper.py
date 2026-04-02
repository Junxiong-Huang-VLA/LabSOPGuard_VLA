from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


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
            }
        )
        if progress_callback:
            progress_callback(idx, total, f"AI 分析中：本地视觉分析已完成 {idx}/{total} 帧。")

    overall = (
        f"已完成 {total} 帧本地视觉分析（云端模型不可用: {reason}）。"
        f" 风险分布：high {risk_counter['high']} / medium {risk_counter['medium']} / "
        f"low {risk_counter['low']} / unknown {risk_counter['unknown']}。"
    )
    return {
        "enabled": True,
        "reason": f"local_fallback_{reason}",
        "analyses": analyses,
        "overall_summary": overall,
        "fallback_mode": "local_visual",
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
    except Exception:
        return _build_local_fallback_result(paths, "client_init_failed", progress_callback)

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
            analyses.append(
                {
                    "image": p.name,
                    "summary": f"云端分析失败，已回退本地视觉分析：{local.get('summary', str(exc))}",
                    "risk_level": str(local.get("risk_level", "unknown") or "unknown"),
                }
            )
        if progress_callback:
            progress_callback(idx, total, f"AI 分析中：已完成 {idx}/{total} 帧。")

    joined = "\n".join(f"- {item['image']}: {item['summary']}" for item in analyses)
    if cloud_failures > 0:
        overall_summary = (
            f"AI 分析完成，共 {len(analyses)} 帧；其中 {cloud_failures} 帧云端调用失败，已自动回退本地视觉分析。\n{joined}"
        )
    else:
        overall_summary = f"AI analyzed {len(analyses)} keyframes.\n{joined}"
    return {
        "enabled": True,
        "reason": "ok" if cloud_failures == 0 else "partial_cloud_failure_with_local_fallback",
        "analyses": analyses,
        "overall_summary": overall_summary,
    }

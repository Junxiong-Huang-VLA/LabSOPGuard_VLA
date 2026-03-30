from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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
) -> Dict[str, Any]:
    paths = [Path(p) for p in image_paths]
    if not paths:
        return {"enabled": False, "reason": "no_keyframes", "analyses": [], "overall_summary": ""}

    client = build_openai_client(base_url=base_url)
    if client is None:
        return {
            "enabled": False,
            "reason": "missing_openai_client_or_api_key(DOUBAO_API_KEY/ARK_API_KEY)",
            "analyses": [
                {
                    "image": p.name,
                    "summary": "AI analysis skipped (missing DOUBAO_API_KEY/ARK_API_KEY or openai package).",
                    "risk_level": "unknown",
                }
                for p in paths
            ],
            "overall_summary": "AI analysis skipped.",
        }

    user_prompt = prompt or (
        "You are analyzing chemistry lab operation frames. "
        "For each image, return a concise safety observation and inferred step."
    )

    analyses: List[Dict[str, Any]] = []
    for p in paths:
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
            analyses.append(
                {
                    "image": p.name,
                    "summary": f"AI call failed: {exc}",
                    "risk_level": "unknown",
                }
            )

    joined = "\n".join(f"- {item['image']}: {item['summary']}" for item in analyses)
    overall_summary = f"AI analyzed {len(analyses)} keyframes.\n{joined}"
    return {
        "enabled": True,
        "reason": "ok",
        "analyses": analyses,
        "overall_summary": overall_summary,
    }

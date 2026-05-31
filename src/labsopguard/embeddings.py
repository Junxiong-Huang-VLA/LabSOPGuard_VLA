from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


logger = logging.getLogger(__name__)


def hash_embedding(text: str, dims: int = 64) -> List[float]:
    vec = [0.0] * dims
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "little") % dims
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vec[index] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _read_qwen_api_key() -> Optional[str]:
    return (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
    )


def _dashscope_base_url() -> str:
    base_url = (
        os.getenv("MATERIAL_EMBEDDING_BASE_URL")
        or os.getenv("DASHSCOPE_API_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
        or os.getenv("QWEN_BASE_URL")
        or "https://dashscope.aliyuncs.com/api/v1"
    )
    return base_url.replace("/compatible-mode/v1", "/api/v1")


@dataclass
class TextEmbeddingProvider:
    mode: str

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class HashEmbeddingProvider(TextEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(mode="hash_embedding_v1")

    def embed(self, text: str) -> List[float]:
        return hash_embedding(text)


class QwenDashScopeEmbeddingProvider(TextEmbeddingProvider):
    def __init__(self, *, api_key: str, model: str, base_url: Optional[str] = None) -> None:
        super().__init__(mode=f"qwen_dashscope:{model}")
        import dashscope  # type: ignore

        self.dashscope = dashscope
        self.dashscope.base_http_api_url = base_url or _dashscope_base_url()
        self.api_key = api_key
        self.model = model
        self.fallback = HashEmbeddingProvider()
        self._fallback_used = False
        self._consecutive_failures = 0
        self._max_failures_before_fallback = 3

    def embed(self, text: str) -> List[float]:
        if self._consecutive_failures >= self._max_failures_before_fallback:
            return self.fallback.embed(text)

        from labsopguard.resilience import RetryConfig, resilient_call

        def _call_embedding():
            response = self.dashscope.TextEmbedding.call(
                api_key=self.api_key,
                model=self.model,
                input=text or " ",
                dimension=int(os.getenv("MATERIAL_EMBEDDING_DIMENSION", "1024")),
                timeout=30,
            )
            embedding = _extract_dashscope_embedding(response)
            return [float(value) for value in embedding]

        result = resilient_call(
            _call_embedding,
            retry_config=RetryConfig(max_retries=2, backoff_factor=1.5, max_backoff=10.0, timeout=30.0),
            fallback=None,
        )

        if result is not None:
            self._consecutive_failures = 0
            return result

        self._consecutive_failures += 1
        self._fallback_used = True
        if not self.mode.endswith(":fallback_hash"):
            self.mode = f"{self.mode}:fallback_hash"
        logger.warning("Qwen embedding failed after retries (failures=%d); using hash fallback", self._consecutive_failures)
        return self.fallback.embed(text)


def _extract_dashscope_embedding(response: Any) -> List[float]:
    output = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
    if isinstance(output, dict):
        embeddings = output.get("embeddings") or []
        if embeddings:
            return list(embeddings[0].get("embedding") or [])
    if hasattr(response, "output") and isinstance(response.output, dict):
        embeddings = response.output.get("embeddings") or []
        if embeddings:
            return list(embeddings[0].get("embedding") or [])
    raise RuntimeError(f"DashScope embedding response missing vector: {response}")


def get_text_embedding_provider() -> TextEmbeddingProvider:
    provider = os.getenv("MATERIAL_EMBEDDING_PROVIDER", "qwen").strip().lower()
    if provider in {"hash", "local_hash"}:
        return HashEmbeddingProvider()

    api_key = _read_qwen_api_key()
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY is not configured; using hash embedding fallback")
        return HashEmbeddingProvider()

    if provider not in {"qwen", "dashscope", "aliyun", "bailian"}:
        return HashEmbeddingProvider()

    model = os.getenv("MATERIAL_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL") or "text-embedding-v4"
    try:
        return QwenDashScopeEmbeddingProvider(
            api_key=api_key,
            model=model,
            base_url=_dashscope_base_url(),
        )
    except Exception as exc:
        logger.warning("Qwen embedding provider initialization failed; using hash fallback: %s", exc)
        return HashEmbeddingProvider()


def embedding_diagnostics() -> dict[str, Any]:
    provider = os.getenv("MATERIAL_EMBEDDING_PROVIDER", "qwen").strip().lower()
    model = os.getenv("MATERIAL_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL") or "text-embedding-v4"
    api_key_configured = bool(_read_qwen_api_key())
    dashscope_installed = importlib.util.find_spec("dashscope") is not None
    fallback_mode = False
    if provider in {"hash", "local_hash"}:
        status = "hash_forced"
        fallback_mode = True
    elif not api_key_configured:
        status = "missing_api_key_hash_fallback"
        fallback_mode = True
    elif not dashscope_installed:
        status = "dashscope_missing_hash_fallback"
        fallback_mode = True
    elif provider in {"qwen", "dashscope", "aliyun", "bailian"}:
        status = "configured"
    else:
        status = f"unsupported_provider_hash_fallback:{provider}"
        fallback_mode = True
    return {
        "qwen_embedding_status": status,
        "current_embedding_model": model,
        "embedding_provider": provider,
        "dashscope_installed": dashscope_installed,
        "api_key_configured": api_key_configured,
        "fallback_mode": fallback_mode,
    }

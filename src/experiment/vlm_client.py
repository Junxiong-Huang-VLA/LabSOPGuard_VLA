"""
DashScope Qwen-VL 集成客户端
使用阿里云 DashScope API 进行视觉语言模型推理

环境变量:
  DASHSCOPE_API_KEY - API Key
  DASHSCOPE_BASE_URL - API Base URL（默认: https://dashscope.aliyuncs.com/compatible-mode/v1）
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# DashScope SDK
try:
    import dashscope  # type: ignore
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class VLMSceneDescription:
    """Qwen-VL 场景描述结果。"""
    description: str
    detected_activities: List[str]
    object_labels: List[str]
    step_indicators: List[str]      # 步骤关键词
    ppe_status: Dict[str, bool]     # PPE 状态
    raw_response: Dict[str, Any]
    confidence: float
    model: str
    inference_time_ms: int


def default_vlm_model() -> str:
    return (
        os.environ.get("KEY_ACTION_VLM_MODEL")
        or os.environ.get("QWEN_VL_MODEL")
        or os.environ.get("VLM_MODEL")
        or "qwen3.6-plus"
    )


class DashScopeVLClient:
    """
    DashScope Qwen-VL 客户端。
    默认使用 2026-05-07 固化的 key-action VLM assist 模型。
    仍可通过 KEY_ACTION_VLM_MODEL/QWEN_VL_MODEL/VLM_MODEL 或构造参数覆盖。
    """

    DEFAULT_MODEL = default_vlm_model()
    VISION_API_URL = "https://dashscope.aliyuncs.com/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        raw_base_url = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL", self.VISION_API_URL
        )
        self.base_url = raw_base_url.replace("/compatible-mode/v1", "/api/v1")
        self.model = model or default_vlm_model()
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is required. "
                "Set it via env var or pass api_key parameter."
            )

        if not DASHSCOPE_AVAILABLE:
            raise ImportError(
                "dashscope Python package is required for Qwen-VL. "
                "Install with: pip install dashscope"
            )

        dashscope.base_http_api_url = self.base_url

    def _encode_image_base64(self, image_path: str) -> str:
        """将图片文件编码为 base64 字符串。"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _is_url(self, path: str) -> bool:
        return path.startswith("http://") or path.startswith("https://")

    def _image_reference(self, image_path: str) -> str:
        if self._is_url(image_path):
            return image_path
        path = Path(image_path)
        if path.exists() and path.is_file():
            mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
            if mime_type.startswith("image/"):
                return f"data:{mime_type};base64,{self._encode_image_base64(str(path))}"
        return f"file://{path.resolve().as_posix()}"

    def _extract_response_text(self, response: Any) -> str:
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

    def describe_scene(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> VLMSceneDescription:
        """
        使用 Qwen-VL 分析单帧图像。

        Args:
            image_path: 图片路径（本地文件或 URL）
            prompt: 自定义提示词
            temperature: 生成温度

        Returns:
            VLMSceneDescription 对象
        """
        import time
        start_ms = int(time.time() * 1000)

        system_prompt = (
            "You are a laboratory experiment analysis assistant. "
            "Analyze the image and respond with a JSON object containing:\n"
            "{\n"
            '  "description": "detailed scene description in Chinese",\n'
            '  "detected_activities": ["list of detected activities"],\n'
            '  "object_labels": ["detected objects and equipment"],\n'
            '  "step_indicators": ["keywords indicating experiment steps"],\n'
            '  "ppe_status": {"gloves": bool, "goggles": bool, "lab_coat": bool},\n'
            '  "confidence": 0.0-1.0\n'
            "}"
        )

        user_prompt = prompt or (
            "请分析这张实验室操作图像：\n"
            "1. 描述当前场景\n"
            '2. 列出检测到的活动（如：吸取液体、加样、离心、移液等）\n'
            '3. 列出检测到的物体和设备\n'
            '4. 列出步骤关键词（如：准备、反应、观察、记录等）\n'
            '5. 判断操作人员是否佩戴 PPE（手套、护目镜、实验服）\n'
            "请以 JSON 格式输出分析结果。"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": self._image_reference(image_path)},
                    {"text": f"{system_prompt}\n\n{user_prompt}"},
                ],
            }
        ]

        response = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        end_ms = int(time.time() * 1000)
        raw = self._extract_response_text(response)
        if not raw:
            raise RuntimeError(
                "DashScope Qwen-VL response missing text: "
                f"{getattr(response, 'code', '')} {getattr(response, 'message', '')}"
            )

        import json as _json
        try:
            parsed = _json.loads(raw)
        except Exception as exc:
            logger.debug("DashScope response is not strict JSON, using raw text fallback: %s", exc)
            parsed = {"description": raw, "confidence": 0.3}

        activities = parsed.get("detected_activities", [])
        if isinstance(activities, str):
            activities = [activities]

        objects = parsed.get("object_labels", [])
        if isinstance(objects, str):
            objects = [objects]

        ppe = parsed.get("ppe_status", {})
        if not isinstance(ppe, dict):
            ppe = {}

        return VLMSceneDescription(
            description=parsed.get("description", ""),
            detected_activities=activities,
            object_labels=objects,
            step_indicators=parsed.get("step_indicators", []),
            ppe_status=ppe,
            raw_response=parsed,
            confidence=float(parsed.get("confidence", 0.7)),
            model=self.model,
            inference_time_ms=end_ms - start_ms,
        )

    def describe_frame_stream(
        self,
        frame_paths: List[str],
        common_prompt: Optional[str] = None,
    ) -> List[VLMSceneDescription]:
        """
        批量分析多个帧（串行调用）。

        Args:
            frame_paths: 帧文件路径列表
            common_prompt: 统一的提示词

        Returns:
            VLMSceneDescription 列表
        """
        results = []
        for path in frame_paths:
            try:
                desc = self.describe_scene(path, prompt=common_prompt)
                results.append(desc)
            except Exception as exc:
                logger.warning("DashScope frame analysis failed for %s: %s", path, exc)
                # 添加 fallback 描述
                results.append(VLMSceneDescription(
                    description="",
                    detected_activities=[],
                    object_labels=[],
                    step_indicators=[],
                    ppe_status={},
                    raw_response={},
                    confidence=0.0,
                    model=self.model,
                    inference_time_ms=0,
                ))
        return results

    def check_health(self) -> bool:
        """检查 API 连接是否正常。"""
        try:
            # 用一个最小的请求测试连接
            import tempfile, numpy as np, cv2
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                # 创建一个纯黑测试图
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                cv2.imwrite(tmp_path, img)

                desc = self.describe_scene(
                    tmp_path,
                    prompt="描述图像内容（简短）",
                    temperature=0.0,
                )
                return True
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError as exc:
                    logger.debug("Failed to remove temporary VLM health image %s: %s", tmp_path, exc)
        except Exception as exc:
            logger.warning("DashScope VLM health check failed: %s", exc)
            return False


def get_vlm_client() -> DashScopeVLClient:
    """获取全局 VLM 客户端（单例模式）。"""
    if not hasattr(get_vlm_client, "_client"):
        get_vlm_client._client = DashScopeVLClient()
    return get_vlm_client._client


def set_vlm_client(client: DashScopeVLClient) -> None:
    """设置全局 VLM 客户端。"""
    get_vlm_client._client = client

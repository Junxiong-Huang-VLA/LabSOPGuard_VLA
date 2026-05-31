from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests


class FeishuConfigError(RuntimeError):
    pass


class FeishuApiError(RuntimeError):
    pass


@dataclass
class FeishuSendResult:
    text_sent: bool = False
    image_sent: bool = False
    image_key: str = ""
    message_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_sent": self.text_sent,
            "image_sent": self.image_sent,
            "image_key": self.image_key,
            "message_ids": self.message_ids,
        }


_TOKEN_LOCK = threading.Lock()
_TOKEN_VALUE = ""
_TOKEN_EXPIRES_AT = 0.0


class FeishuNotifier:
    """Small Feishu OpenAPI client for sending local JPEG bytes to one chat."""

    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        receive_id: str,
        receive_id_type: str = "chat_id",
        api_base: str = "https://open.feishu.cn/open-apis",
        timeout_sec: float = 20.0,
        max_retries: int = 5,
        trust_env_proxy: bool = False,
    ) -> None:
        self.app_id = app_id.strip()
        self.app_secret = app_secret.strip()
        self.receive_id = receive_id.strip()
        self.receive_id_type = receive_id_type.strip() or "chat_id"
        self.api_base = api_base.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.max_retries = max(1, int(max_retries))
        self._session = requests.Session()
        # Local Windows proxy settings are often stale during demos; Feishu should
        # use a direct HTTPS connection unless FEISHU_TRUST_ENV_PROXY=1 is set.
        self._session.trust_env = bool(trust_env_proxy)

    @classmethod
    def from_env(cls) -> "FeishuNotifier":
        return cls(
            app_id=os.getenv("FEISHU_APP_ID", ""),
            app_secret=os.getenv("FEISHU_APP_SECRET", ""),
            receive_id=os.getenv("FEISHU_RECEIVE_ID", ""),
            receive_id_type=os.getenv("FEISHU_RECEIVE_ID_TYPE", "chat_id"),
            api_base=os.getenv("FEISHU_API_BASE", "https://open.feishu.cn/open-apis"),
            timeout_sec=float(os.getenv("FEISHU_TIMEOUT_SEC", "20")),
            max_retries=int(os.getenv("FEISHU_RETRIES", "5")),
            trust_env_proxy=os.getenv("FEISHU_TRUST_ENV_PROXY", "0").strip().lower()
            in {"1", "true", "yes", "on"},
        )

    @classmethod
    def from_env_for_receive(cls, *, receive_id: str, receive_id_type: str) -> "FeishuNotifier":
        return cls(
            app_id=os.getenv("FEISHU_APP_ID", ""),
            app_secret=os.getenv("FEISHU_APP_SECRET", ""),
            receive_id=receive_id,
            receive_id_type=receive_id_type,
            api_base=os.getenv("FEISHU_API_BASE", "https://open.feishu.cn/open-apis"),
            timeout_sec=float(os.getenv("FEISHU_TIMEOUT_SEC", "20")),
            max_retries=int(os.getenv("FEISHU_RETRIES", "5")),
            trust_env_proxy=os.getenv("FEISHU_TRUST_ENV_PROXY", "0").strip().lower()
            in {"1", "true", "yes", "on"},
        )

    def validate_config(self) -> None:
        missing = []
        if not self.app_id:
            missing.append("FEISHU_APP_ID")
        if not self.app_secret:
            missing.append("FEISHU_APP_SECRET")
        if not self.receive_id:
            missing.append("FEISHU_RECEIVE_ID")
        if missing:
            raise FeishuConfigError("Feishu is not configured; set " + ", ".join(missing))

    def send_text_and_image(
        self,
        *,
        text: str,
        image_bytes: bytes,
        filename: str = "snapshot.jpg",
    ) -> FeishuSendResult:
        self.validate_config()
        result = FeishuSendResult()
        text_msg = self.send_text(text)
        result.text_sent = True
        if text_msg:
            result.message_ids.append(text_msg)
        image_key = self.upload_image(image_bytes, filename=filename)
        result.image_key = image_key
        image_msg = self.send_image(image_key)
        result.image_sent = True
        if image_msg:
            result.message_ids.append(image_msg)
        return result

    def send_text(self, text: str) -> str:
        return self._send_message("text", {"text": text})

    def send_image(self, image_key: str) -> str:
        return self._send_message("image", {"image_key": image_key})

    def upload_image(
        self,
        image_bytes: bytes,
        *,
        filename: str = "snapshot.jpg",
        content_type: str = "image/jpeg",
    ) -> str:
        if not image_bytes:
            raise FeishuApiError("Cannot upload empty image")
        token = self._tenant_access_token()
        url = f"{self.api_base}/im/v1/images"
        headers = {"Authorization": f"Bearer {token}"}
        data = {"image_type": "message"}
        files = {"image": (filename, image_bytes, content_type)}
        resp = self._post(url, headers=headers, data=data, files=files)
        payload = self._parse_response(resp)
        image_key = str((payload.get("data") or {}).get("image_key") or "")
        if not image_key:
            raise FeishuApiError(f"Feishu image upload returned no image_key: {payload}")
        return image_key

    def _send_message(self, msg_type: str, content: dict[str, Any]) -> str:
        token = self._tenant_access_token()
        url = f"{self.api_base}/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        body = {
            "receive_id": self.receive_id,
            "msg_type": msg_type,
            "content": json.dumps(content, ensure_ascii=False),
        }
        resp = self._post(
            url,
            params={"receive_id_type": self.receive_id_type},
            headers=headers,
            json=body,
        )
        payload = self._parse_response(resp)
        return str((payload.get("data") or {}).get("message_id") or "")

    def _tenant_access_token(self) -> str:
        global _TOKEN_EXPIRES_AT, _TOKEN_VALUE
        now = time.time()
        with _TOKEN_LOCK:
            if _TOKEN_VALUE and now < _TOKEN_EXPIRES_AT:
                return _TOKEN_VALUE
            url = f"{self.api_base}/auth/v3/tenant_access_token/internal"
            resp = self._post(
                url,
                json={"app_id": self.app_id, "app_secret": self.app_secret},
            )
            payload = self._parse_response(resp)
            token = str(payload.get("tenant_access_token") or "")
            if not token:
                raise FeishuApiError(f"Feishu token response missing tenant_access_token: {payload}")
            expire = float(payload.get("expire") or 7200)
            _TOKEN_VALUE = token
            _TOKEN_EXPIRES_AT = now + max(60.0, expire - 120.0)
            return _TOKEN_VALUE

    def _post(self, url: str, **kwargs: Any) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._session.post(url, timeout=self.timeout_sec, **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(0.4 * (attempt + 1))
        raise FeishuApiError(f"Feishu request failed: {type(last_exc).__name__}: {last_exc}") from last_exc

    @staticmethod
    def _parse_response(resp: requests.Response) -> dict[str, Any]:
        try:
            payload = resp.json()
        except ValueError as exc:
            raise FeishuApiError(f"Feishu returned non-JSON response: HTTP {resp.status_code}") from exc
        if resp.status_code >= 400:
            raise FeishuApiError(f"Feishu HTTP {resp.status_code}: {payload}")
        code = payload.get("code", 0)
        if code not in (0, "0"):
            raise FeishuApiError(f"Feishu API error: {payload}")
        return payload

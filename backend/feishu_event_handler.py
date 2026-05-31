from __future__ import annotations

import base64
import hashlib
import ast
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


class FeishuEventError(RuntimeError):
    pass


@dataclass
class FeishuInboundMessage:
    event_id: str
    event_type: str
    message_id: str
    chat_id: str
    chat_type: str
    sender_open_id: str
    message_type: str
    text: str
    create_time_iso: Optional[str]
    raw: dict[str, Any]


@dataclass
class ParsedFeishuStepCommand:
    command: str
    experiment_id: Optional[str]
    step_text: str
    message_sent_at: Optional[str] = None
    window_before_sec: Optional[float] = None
    window_after_sec: Optional[float] = None
    limit: Optional[int] = None


def decode_feishu_event_payload(payload: Mapping[str, Any], *, encrypt_key: Optional[str] = None) -> dict[str, Any]:
    """Decode a Feishu callback payload, including optional encrypted callbacks."""
    if "encrypt" not in payload:
        return dict(payload)
    if not encrypt_key:
        raise FeishuEventError("Encrypted Feishu callback received, but FEISHU_ENCRYPT_KEY is not configured")
    return _decrypt_event_payload(str(payload.get("encrypt") or ""), encrypt_key=encrypt_key)


def feishu_url_verification_response(payload: Mapping[str, Any], *, expected_token: Optional[str] = None) -> Optional[dict[str, str]]:
    """Return challenge response for Feishu URL verification, or None for regular events."""
    challenge = payload.get("challenge")
    is_verification = payload.get("type") == "url_verification" or bool(challenge and "event" not in payload)
    if not is_verification:
        return None
    _validate_feishu_token(payload, expected_token=expected_token)
    return {"challenge": str(challenge or "")}


def parse_feishu_inbound_message(payload: Mapping[str, Any], *, expected_token: Optional[str] = None) -> Optional[FeishuInboundMessage]:
    """Parse im.message.receive_v1 into a normalized text message."""
    _validate_feishu_token(payload, expected_token=expected_token)
    header = payload.get("header") if isinstance(payload.get("header"), Mapping) else {}
    event = payload.get("event") if isinstance(payload.get("event"), Mapping) else {}
    event_type = str(header.get("event_type") or payload.get("type") or "")
    if event_type and event_type != "im.message.receive_v1":
        return None
    message = event.get("message") if isinstance(event.get("message"), Mapping) else {}
    sender = event.get("sender") if isinstance(event.get("sender"), Mapping) else {}
    sender_id = sender.get("sender_id") if isinstance(sender.get("sender_id"), Mapping) else {}
    message_type = str(message.get("message_type") or "")
    if message_type and message_type != "text":
        return FeishuInboundMessage(
            event_id=str(header.get("event_id") or ""),
            event_type=event_type or "im.message.receive_v1",
            message_id=str(message.get("message_id") or ""),
            chat_id=str(message.get("chat_id") or ""),
            chat_type=str(message.get("chat_type") or ""),
            sender_open_id=str(sender_id.get("open_id") or ""),
            message_type=message_type,
            text="",
            create_time_iso=_feishu_ms_to_iso(message.get("create_time") or header.get("create_time")),
            raw=dict(payload),
        )
    text = _extract_text_content(message.get("content"))
    text = _strip_feishu_mentions(text)
    return FeishuInboundMessage(
        event_id=str(header.get("event_id") or ""),
        event_type=event_type or "im.message.receive_v1",
        message_id=str(message.get("message_id") or ""),
        chat_id=str(message.get("chat_id") or ""),
        chat_type=str(message.get("chat_type") or ""),
        sender_open_id=str(sender_id.get("open_id") or ""),
        message_type=message_type or "text",
        text=text.strip(),
        create_time_iso=_feishu_ms_to_iso(message.get("create_time") or header.get("create_time")),
        raw=dict(payload),
    )


def parse_step_command(text: str) -> ParsedFeishuStepCommand:
    """Parse a Feishu text message into bind/help/query commands."""
    clean = _strip_feishu_mentions(text).strip()
    if not clean:
        return ParsedFeishuStepCommand(command="help", experiment_id=None, step_text="")
    json_command = _parse_jsonish_step_command(clean)
    if json_command is not None:
        return json_command
    compact = re.sub(r"\s+", " ", clean).strip()
    lowered = compact.lower()
    if lowered in {"help", "帮助", "使用帮助", "?"}:
        return ParsedFeishuStepCommand(command="help", experiment_id=None, step_text="")

    bind_match = re.search(r"^(?:绑定实验|bind(?:\s+experiment)?|绑定)\s*[:：]?\s*([A-Za-z0-9._-]{2,128})\s*$", compact, re.I)
    if bind_match:
        return ParsedFeishuStepCommand(command="bind", experiment_id=bind_match.group(1), step_text="")

    experiment_id = None
    patterns = [
        r"(?:实验ID|实验id|experiment_id|experiment id|exp_id|exp id)\s*[:：=]\s*([A-Za-z0-9._-]{2,128})",
        r"(?:实验|experiment|exp)\s*[:：=]\s*([A-Za-z0-9._-]{2,128})",
    ]
    step_text = clean
    for pattern in patterns:
        match = re.search(pattern, step_text, re.I)
        if match:
            experiment_id = match.group(1)
            step_text = (step_text[: match.start()] + step_text[match.end() :]).strip()
            break
    step_text = re.sub(r"^(?:步骤|step|检查|核验)\s*[:：]\s*", "", step_text, flags=re.I).strip()
    return ParsedFeishuStepCommand(command="query", experiment_id=experiment_id, step_text=step_text or clean)


def _parse_jsonish_step_command(text: str) -> Optional[ParsedFeishuStepCommand]:
    payload = _extract_jsonish_mapping(text) or _parse_key_value_mapping(text)
    if not payload:
        return None

    command = str(payload.get("command") or payload.get("type") or "").strip().lower()
    experiment_id = _clean_experiment_id(
        payload.get("experiment_id")
        or payload.get("experiment id")
        or payload.get("exp_id")
        or payload.get("exp id")
        or payload.get("experiment")
        or payload.get("exp")
    )
    step_text = _clean_text(
        payload.get("step_text")
        or payload.get("query")
        or payload.get("question")
        or payload.get("step")
        or payload.get("text")
    )
    if command in {"help", "?"}:
        return ParsedFeishuStepCommand(command="help", experiment_id=None, step_text="")
    if command in {"bind", "bind_experiment"} and experiment_id:
        return ParsedFeishuStepCommand(command="bind", experiment_id=experiment_id, step_text="")

    has_query_shape = bool(
        experiment_id
        or step_text
        or payload.get("message_sent_at")
        or payload.get("sent_at")
        or payload.get("window")
        or payload.get("limit") is not None
    )
    if not has_query_shape:
        return None

    before, after = _jsonish_window_seconds(payload)
    return ParsedFeishuStepCommand(
        command="query",
        experiment_id=experiment_id,
        step_text=step_text,
        message_sent_at=_clean_text(payload.get("message_sent_at") or payload.get("sent_at")),
        window_before_sec=before,
        window_after_sec=after,
        limit=_optional_int(payload.get("limit")),
    )


def _extract_jsonish_mapping(text: str) -> dict[str, Any]:
    clean = _strip_json_code_fence(text)
    candidates = [clean]
    match = re.search(r"\{.*\}", clean, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(candidate)
            except Exception:
                continue
            if isinstance(parsed, Mapping):
                return dict(parsed)
    return {}


def _strip_json_code_fence(text: str) -> str:
    clean = str(text or "").strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.I)
    clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def _parse_key_value_mapping(text: str) -> dict[str, Any]:
    keys = {
        "experiment_id",
        "experiment id",
        "exp_id",
        "exp id",
        "experiment",
        "exp",
        "query",
        "question",
        "step_text",
        "step",
        "message_sent_at",
        "sent_at",
        "window",
        "window_before_sec",
        "window_after_sec",
        "before_sec",
        "after_sec",
        "limit",
    }
    parts = re.split(r"[\r\n;]+", str(text or ""))
    if len(parts) == 1:
        parts = re.split(
            r",\s*(?=(?:experiment_id|experiment id|exp_id|exp id|experiment|exp|query|question|step_text|step|message_sent_at|sent_at|window|window_before_sec|window_after_sec|before_sec|after_sec|limit)\s*[:=])",
            str(text or ""),
            flags=re.I,
        )
    payload: dict[str, Any] = {}
    for part in parts:
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_ ]{1,40})\s*[:=]\s*(.*?)\s*$", part, flags=re.S)
        if not match:
            continue
        key = re.sub(r"\s+", " ", match.group(1).strip().lower())
        if key not in keys:
            continue
        value = match.group(2).strip().strip(",")
        payload[key] = _unquote_jsonish_scalar(value)
    if not any(key in payload for key in ("query", "question", "step_text", "step")):
        return {}
    return payload


def _unquote_jsonish_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if text[0:1] in {"'", '"'} and text[-1:] == text[0]:
        return text[1:-1]
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            continue
    return text


def _jsonish_window_seconds(payload: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
    before = _optional_float(payload.get("window_before_sec") or payload.get("before_sec"))
    after = _optional_float(payload.get("window_after_sec") or payload.get("after_sec"))
    window = payload.get("window")
    if isinstance(window, Mapping):
        if before is None:
            before = _optional_float(window.get("before_sec") or window.get("before") or window.get("window_before_sec"))
        if after is None:
            after = _optional_float(window.get("after_sec") or window.get("after") or window.get("window_after_sec"))
    elif isinstance(window, (list, tuple)) and len(window) >= 2:
        before = before if before is not None else _optional_float(window[0])
        after = after if after is not None else _optional_float(window[1])
    elif window not in (None, ""):
        value = _optional_float(window)
        before = before if before is not None else value
        after = after if after is not None else value
    return before, after


def _clean_experiment_id(value: Any) -> Optional[str]:
    text = _clean_text(value)
    if not text:
        return None
    match = re.search(r"[A-Za-z0-9._-]{2,128}", text)
    return match.group(0) if match else None


def _clean_text(value: Any) -> str:
    if value in (None, [], {}):
        return ""
    return str(value).strip()


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(1, min(number, 50))


def load_feishu_chat_bindings(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {"schema_version": "feishu_chat_bindings.v1", "bindings": {}}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": "feishu_chat_bindings.v1", "bindings": {}}
    if not isinstance(payload, dict):
        return {"schema_version": "feishu_chat_bindings.v1", "bindings": {}}
    payload.setdefault("schema_version", "feishu_chat_bindings.v1")
    payload.setdefault("bindings", {})
    return payload


def save_feishu_chat_binding(path: str | Path, *, chat_id: str, experiment_id: str) -> dict[str, Any]:
    file_path = Path(path)
    payload = load_feishu_chat_bindings(file_path)
    bindings = payload.setdefault("bindings", {})
    bindings[str(chat_id)] = {
        "experiment_id": str(experiment_id),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def get_bound_experiment_id(path: str | Path, *, chat_id: str) -> Optional[str]:
    payload = load_feishu_chat_bindings(path)
    row = (payload.get("bindings") or {}).get(str(chat_id))
    if not isinstance(row, Mapping):
        return None
    value = str(row.get("experiment_id") or "").strip()
    return value or None


def build_feishu_help_text(default_experiment_id: Optional[str] = None) -> str:
    lines = [
        "【LabSOPGuard 使用方式】",
        "1. 绑定当前会话：绑定实验 <实验ID>",
        "2. 核验步骤：检查试剂瓶归位是否正确",
        "3. 也可以不绑定，直接发送：实验ID: <实验ID>\\n检查试剂瓶归位是否正确",
    ]
    if default_experiment_id:
        lines.append(f"当前默认实验：{default_experiment_id}")
    return "\n".join(lines)


def _validate_feishu_token(payload: Mapping[str, Any], *, expected_token: Optional[str]) -> None:
    expected = str(expected_token or "").strip()
    if not expected:
        return
    header = payload.get("header") if isinstance(payload.get("header"), Mapping) else {}
    actual = str(payload.get("token") or header.get("token") or "").strip()
    if actual != expected:
        raise FeishuEventError("Invalid Feishu verification token")


def _extract_text_content(content: Any) -> str:
    if isinstance(content, Mapping):
        return str(content.get("text") or "")
    raw = str(content or "").strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, Mapping):
        return str(parsed.get("text") or "")
    return raw


def _strip_feishu_mentions(text: str) -> str:
    text = re.sub(r"<at[^>]*>.*?</at>", " ", str(text or ""), flags=re.I | re.S)
    text = re.sub(r"@\S+", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _feishu_ms_to_iso(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 10_000_000_000:
        number = number / 1000.0
    return datetime.fromtimestamp(number, tz=timezone.utc).isoformat()


def _decrypt_event_payload(encrypted: str, *, encrypt_key: str) -> dict[str, Any]:
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.padding import PKCS7
    except Exception as exc:  # pragma: no cover - dependency exists in project env.
        raise FeishuEventError("cryptography is required to decrypt Feishu callbacks") from exc
    try:
        raw = base64.b64decode(encrypted)
        key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
        iv, ciphertext = raw[:16], raw[16:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()
        unpadder = PKCS7(128).unpadder()
        data = unpadder.update(padded) + unpadder.finalize()
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise FeishuEventError("Failed to decrypt Feishu callback payload") from exc
    if not isinstance(payload, dict):
        raise FeishuEventError("Decrypted Feishu callback payload is not an object")
    return payload

"""PTZ 浜轰綋璺熼殢妯″潡鍚庣璺敱

鎶?D:/LabEmbodiedVLA/ptz_tracker 鍖呭寘瑁呮垚涓€涓湪绾挎湇鍔★細
- 鍚庡彴绾跨▼璇绘憚鍍忓ご -> 浜轰綋妫€娴?+ 鍚堣妫€娴?-> PID 璺熻釜 -> MQTT 涓嬪彂浜戝彴鎸囦护
- 瀵瑰鏆撮湶 MJPEG 鎺ㄦ祦 + REST 鎺у埗鎺ュ彛
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ptz_tracker 鍖呬綅浜庝粨搴撴牴鐩綍锛圠abSOPGuard 鐨勪笂涓€绾э級
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PTZ_PPE_KEYFRAME_DIR = _PROJECT_ROOT / "outputs" / "ptz_ppe_keyframes"
# Force Ultralytics to use a repo-local writable settings directory.
_YOLO_CONFIG_DIR = _PROJECT_ROOT / "outputs" / "ultralytics"
_YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_YOLO_CONFIG_DIR))

from ptz_tracker.config import AppConfig  # noqa: E402
from ptz_tracker.detector import ComplianceResult, Detector, PersonDetection  # noqa: E402
from ptz_tracker.mqtt_ptz import MqttPtzController  # noqa: E402
from ptz_tracker.tracker import TrackingEngine, TrackState  # noqa: E402
from ptz_tracker.video_source import create_video_source  # noqa: E402

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ptz-tracker", tags=["ptz-tracker"])


@dataclass
class _Violation:
    ts: float
    types: list[str]
    lab_coat_conf: float
    gloves_conf: float
    event_type: str = "violation"
    keyframe_path: str = ""

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "event_type": self.event_type,
            "types": self.types,
            "lab_coat_conf": round(self.lab_coat_conf, 3),
            "gloves_conf": round(self.gloves_conf, 3),
            "keyframe_path": self.keyframe_path,
        }


@dataclass
class _SharedFrame:
    jpeg: Optional[bytes] = None
    persons: list = field(default_factory=list)
    compliance: ComplianceResult = field(default_factory=ComplianceResult)
    state: str = TrackState.IDLE.value
    ptz_pitch: int = 90
    ptz_yaw: int = 90
    ptz_speed: float = 1.0
    mqtt_connected: bool = False
    fps: float = 0.0
    timestamp: float = 0.0
    error: Optional[str] = None
    alert_active: bool = False
    alert_reasons: list = field(default_factory=list)
    violations: list = field(default_factory=list)  # 鏈€杩?20 鏉?

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

PPE_BOX_LABELS = {
    2: "gloved_hand",
    3: "lab_coat",
}

PPE_REASON_LABELS_ZH = {
    "no_lab_coat": "未穿实验服",
    "no_gloves": "未戴手套",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _ptz_feishu_alert_enabled() -> bool:
    return _env_bool("PTZ_FEISHU_ALERT_ENABLED", True)


def _build_ppe_prompt(types: list[str]) -> str:
    missing = set(types)
    if {"no_lab_coat", "no_gloves"}.issubset(missing):
        return "检测到实验员未穿实验服且未戴手套，请穿戴好实验服和手套。"
    if "no_lab_coat" in missing:
        return "检测到实验员未穿实验服，请穿好实验服。"
    if "no_gloves" in missing:
        return "检测到实验员未戴手套，请戴好手套。"
    return "检测到实验员 PPE 穿戴异常，请检查实验服和手套。"


def _format_ppe_types_zh(types: list[str]) -> str:
    return "、".join(PPE_REASON_LABELS_ZH.get(t, t) for t in types) or "无"


def _split_host_port(endpoint: str) -> tuple[str, str]:
    endpoint = endpoint.strip()
    if not endpoint:
        return "", ""
    if endpoint.startswith("[") and "]:" in endpoint:
        host, _, port = endpoint[1:].rpartition("]:")
        return host, port
    if endpoint.count(":") >= 2 and "." not in endpoint:
        # IPv6 without brackets in netstat output.
        host, _, port = endpoint.rpartition(":")
        return host, port
    host, _, port = endpoint.rpartition(":")
    return host, port


def _get_mqtt_remote_clients(port: int) -> list[str]:
    """Best-effort netstat parsing for remote TCP clients connected to broker port."""
    try:
        out = subprocess.run(
            ["netstat", "-ano"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        ).stdout
    except Exception:
        return []

    remote_clients: list[str] = []
    for line in out.splitlines():
        if "ESTABLISHED" not in line.upper():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        local = parts[1]
        remote = parts[2]
        local_host, local_port = _split_host_port(local)
        remote_host, _ = _split_host_port(remote)
        if local_port != str(port):
            continue
        if remote_host in {"127.0.0.1", "::1", "localhost"}:
            continue
        if not remote_host:
            continue
        if remote_host not in remote_clients:
            remote_clients.append(remote_host)
    return remote_clients


def _draw_overlay(
    frame: np.ndarray,
    persons: list[PersonDetection],
    target: PersonDetection | None,
    comp: ComplianceResult,
    state: str,
    ptz_pitch: int,
    ptz_yaw: int,
    fps: float,
    mqtt_ok: bool,
    alert_active: bool = False,
    alert_reasons: list | None = None,
) -> np.ndarray:
    frame = frame.copy()
    h, w = frame.shape[:2]

    # crosshair
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), COLOR_WHITE, 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), COLOR_WHITE, 1)

    # person boxes
    for p in persons:
        is_target = target is not None and p.center == target.center
        color = COLOR_GREEN if is_target else COLOR_CYAN
        thickness = 2 if is_target else 1
        x1, y1, x2, y2 = p.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"person {p.confidence:.2f}"
        if is_target:
            label = "[TARGET] " + label
            cv2.drawMarker(frame, p.center, COLOR_GREEN, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for item in comp.detail_boxes:
        cls_id = int(item.get("cls", -1))
        if cls_id not in PPE_BOX_LABELS:
            continue
        x1, y1, x2, y2 = [int(v) for v in item.get("bbox", [0, 0, 0, 0])]
        conf = float(item.get("conf", 0.0))
        color = COLOR_GREEN if cls_id == 3 else COLOR_YELLOW
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{PPE_BOX_LABELS[cls_id]} {conf:.2f}",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    # status HUD
    state_color = {
        "tracking": COLOR_GREEN,
        "lost": COLOR_RED,
        "manual": COLOR_YELLOW,
    }.get(state, COLOR_WHITE)
    cv2.putText(frame, f"State: {state.upper()}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
    cv2.putText(frame, f"PTZ P:{ptz_pitch} Y:{ptz_yaw}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    cv2.putText(frame, f"MQTT: {'ON' if mqtt_ok else 'OFF'}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                COLOR_GREEN if mqtt_ok else COLOR_RED, 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # compliance HUD
    y0 = 120
    glove_color = COLOR_GREEN if comp.has_gloves else COLOR_RED
    cv2.putText(frame,
                f"Gloves: {'YES' if comp.has_gloves else 'NO'} ({comp.gloves_conf:.2f})",
                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, glove_color, 2)
    cv2.putText(frame,
                "Lab Coat: PAUSED",
                (10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
    status_text = "GLOVES OK" if comp.has_gloves else "NO GLOVES"
    status_color = COLOR_GREEN if comp.compliant else COLOR_RED
    cv2.putText(frame, status_text, (10, y0 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # 杩濊鍛婅锛氬叏灞忕孩妗?+ 椤堕儴绾㈡潯锛堝熀浜庢椂闂撮棯鐑侊級
    if alert_active:
        blink_on = int(time.monotonic() * 2) % 2 == 0
        if blink_on:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_RED, 6)
        banner_h = 36
        banner = frame[0:banner_h, :].copy()
        cv2.rectangle(frame, (0, 0), (w, banner_h), COLOR_RED, -1)
        cv2.addWeighted(banner, 0.25, frame[0:banner_h, :], 0.75, 0, frame[0:banner_h, :])
        msg = "PPE VIOLATION: " + ", ".join(alert_reasons or [])
        cv2.putText(frame, msg, (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

    return frame


class PtzTrackerService:
    """Background service for camera, detector, PTZ control, and alerts."""

    def __init__(self) -> None:
        self._cfg = AppConfig()
        self._source = None
        self._detector: Optional[Detector] = None
        self._engine: Optional[TrackingEngine] = None
        self._ptz: Optional[MqttPtzController] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._shared = _SharedFrame()
        self._last_compliance = ComplianceResult()
        self._last_compliance_time = 0.0
        self._frame_count = 0
        self._fps_time = time.monotonic()

        self._violation_streak = 0
        self._violation_threshold = 3
        self._alert_active = False
        self._alert_reasons: list[str] = []
        self._violations: list[_Violation] = []
        self._last_alert_emit_ts = 0.0
        self._last_alert_emit_by_signature: dict[tuple[str, ...], float] = {}
        self._alert_emit_cooldown = 60.0
        self._alert_clear_frames = 0
        self._alert_clear_threshold = 3
        self._last_remote_client_check_ts = 0.0

        self._started = False
        self._start_error: Optional[str] = None

    @property
    def started(self) -> bool:
        return self._started

    def ensure_started(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            try:
                self._source = create_video_source(self._cfg.video)
                if not self._source.open():
                    raise RuntimeError("Failed to open video source")

                self._detector = Detector(self._cfg.detector)
                self._detector.load()

                self._engine = TrackingEngine(self._cfg.pid, self._cfg.tracking)
                self._ptz = MqttPtzController(self._cfg.mqtt, self._cfg.ptz)
                self._ptz.connect()

                self._stop_evt.clear()
                self._thread = threading.Thread(
                    target=self._run_loop, name="ptz-tracker", daemon=True
                )
                self._thread.start()
                self._started = True
                self._start_error = None
                logger.info(
                    "PTZ tracker service started: mode=%s camera_index=%s wvd_camera=%s",
                    self._cfg.video.mode,
                    self._cfg.video.camera_index,
                    self._cfg.video.wvd_camera,
                )
            except Exception as e:
                self._start_error = str(e)
                self._started = False
                logger.exception("PTZ tracker start failed")
                # 宸叉墦寮€鐨勮祫婧愬敖閲忛噴鏀?                self._release_internal()
                raise

    def shutdown(self) -> None:
        if not self._started and self._thread is None:
            return
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._release_internal()
        self._started = False
        logger.info("PTZ tracker service stopped")

    def _release_internal(self) -> None:
        try:
            if self._engine is not None:
                self._engine.stop_tracking()
        except Exception:
            pass
        try:
            if self._ptz is not None:
                self._ptz.stop()
                self._ptz.disconnect()
        except Exception:
            pass
        try:
            if self._source is not None:
                self._source.release()
        except Exception:
            pass
        self._engine = None
        self._ptz = None
        self._source = None
        self._detector = None

    # ---- Control commands ----
    def start_tracking(self) -> None:
        self.ensure_started()
        assert self._engine is not None and self._ptz is not None
        if self._cfg.tracking.require_mqtt_connected and not self._ptz.connected:
            raise RuntimeError(
                "MQTT is not connected. Start MQTT broker and verify PTZ client connectivity."
            )
        mqtt_remote_clients = _get_mqtt_remote_clients(self._cfg.mqtt.port)
        if self._cfg.tracking.require_remote_mqtt_client and not mqtt_remote_clients:
            raise RuntimeError(
                "No remote MQTT client connected to broker. Check ESP32/robot network and firewall."
            )
        self._engine.start_tracking()

    def stop_tracking(self) -> None:
        if self._engine is not None:
            self._engine.stop_tracking()
        if self._ptz is not None:
            self._ptz.stop()

    def center(self) -> None:
        if self._engine is not None:
            self._engine.stop_tracking()
        if self._ptz is not None:
            self._ptz.center()

    def enter_manual(self) -> None:
        if self._engine is not None:
            self._engine.enter_manual()

    def move(self, direction: str) -> None:
        self.ensure_started()
        assert self._ptz is not None and self._engine is not None
        self._engine.enter_manual()
        d = direction.lower().strip()
        if d == "up":
            self._ptz.step_up()
        elif d == "down":
            self._ptz.step_down()
        elif d == "left":
            self._ptz.step_left()
        elif d == "right":
            self._ptz.step_right()
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def set_speed(self, speed: float) -> None:
        self.ensure_started()
        assert self._ptz is not None
        self._ptz.speed = speed

    def status(self) -> dict:
        with self._lock:
            s = self._shared
            mqtt_remote_clients = _get_mqtt_remote_clients(self._cfg.mqtt.port)
            mqtt_connected = self._ptz.connected if self._ptz is not None else s.mqtt_connected
            ptz_pitch = self._ptz.current_pitch if self._ptz is not None else s.ptz_pitch
            ptz_yaw = self._ptz.current_yaw if self._ptz is not None else s.ptz_yaw
            ptz_speed = self._ptz.speed if self._ptz is not None else s.ptz_speed
            return {
                "started": self._started,
                "start_error": self._start_error,
                "state": s.state,
                "fps": round(s.fps, 2),
                "mqtt_connected": mqtt_connected,
                "mqtt_broker": self._cfg.mqtt.broker,
                "mqtt_port": self._cfg.mqtt.port,
                "mqtt_topic": self._cfg.mqtt.topic,
                "mqtt_remote_client_count": len(mqtt_remote_clients),
                "mqtt_remote_clients": mqtt_remote_clients,
                "tracking_require_mqtt_connected": self._cfg.tracking.require_mqtt_connected,
                "tracking_require_remote_mqtt_client": self._cfg.tracking.require_remote_mqtt_client,
                "ptz_pitch": ptz_pitch,
                "ptz_yaw": ptz_yaw,
                "ptz_speed": ptz_speed,
                "persons_count": len(s.persons),
                "target_center": self._engine.target.center if self._engine and self._engine.target else None,
                "compliance": {
                    "mode": "gloves_only",
                    "has_lab_coat": s.compliance.has_lab_coat,
                    "has_gloves": s.compliance.has_gloves,
                    "lab_coat_conf": round(s.compliance.lab_coat_conf, 3),
                    "gloves_conf": round(s.compliance.gloves_conf, 3),
                    "compliant": s.compliance.compliant,
                    "glove_conf_threshold": round(self._cfg.detector.glove_conf_threshold, 3),
                    "detect_lab_coat": self._cfg.detector.detect_lab_coat,
                },
                "alert": {
                    "active": s.alert_active,
                    "reasons": s.alert_reasons,
                    "cooldown_sec": self._alert_emit_cooldown,
                    "feishu_alert_enabled": _ptz_feishu_alert_enabled(),
                },
                "keyframe_dir": str(PTZ_PPE_KEYFRAME_DIR),
                "video_source": {
                    "source_id": self.video_source_id(),
                    "mode": self._cfg.video.mode,
                    "camera_index": self._cfg.video.camera_index,
                    "wvd_camera": self._cfg.video.wvd_camera,
                },
                "violations": s.violations,
                "error": s.error,
                "timestamp": s.timestamp,
            }

    # ---- Alert state ----
    def _update_alert_state(self, comp: ComplianceResult, wall_ts: float) -> list[_Violation]:
        events: list[_Violation] = []
        reasons: list[str] = []
        if not comp.has_lab_coat:
            reasons.append("no_lab_coat")
        if not comp.has_gloves:
            reasons.append("no_gloves")

        if reasons:
            self._violation_streak += 1
            self._alert_clear_frames = 0
            if self._violation_streak >= self._violation_threshold:
                self._alert_active = True
                self._alert_reasons = reasons
                signature = tuple(sorted(reasons))
                last_emit = self._last_alert_emit_by_signature.get(signature, 0.0)
                if wall_ts - last_emit >= self._alert_emit_cooldown:
                    v = _Violation(
                        ts=wall_ts,
                        types=list(reasons),
                        lab_coat_conf=comp.lab_coat_conf,
                        gloves_conf=comp.gloves_conf,
                        event_type="violation",
                    )
                    self._violations.append(v)
                    if len(self._violations) > 50:
                        self._violations = self._violations[-50:]
                    self._last_alert_emit_ts = wall_ts
                    self._last_alert_emit_by_signature[signature] = wall_ts
                    events.append(v)
        else:
            self._violation_streak = 0
            self._alert_clear_frames += 1
            if self._alert_active and self._alert_clear_frames >= self._alert_clear_threshold:
                self._alert_active = False
                self._alert_reasons = []
                v = _Violation(
                    ts=wall_ts,
                    types=[],
                    lab_coat_conf=comp.lab_coat_conf,
                    gloves_conf=comp.gloves_conf,
                    event_type="recovered",
                )
                self._violations.append(v)
                if len(self._violations) > 50:
                    self._violations = self._violations[-50:]
                events.append(v)
        return events

    @staticmethod
    def _has_actor_signal(persons: list[PersonDetection], comp: ComplianceResult) -> bool:
        if persons:
            return True
        return any(int(item.get("cls", -1)) in PPE_BOX_LABELS for item in comp.detail_boxes)

    def _publish_alert(self, v: _Violation, image_bytes: bytes | None = None) -> None:
        if v.event_type == "violation" and image_bytes:
            self._save_violation_keyframe(v, image_bytes)
        self._publish_mqtt_alert(v)
        if v.event_type == "violation":
            self._publish_feishu_alert(v, image_bytes=image_bytes)

    def _save_violation_keyframe(self, v: _Violation, image_bytes: bytes) -> None:
        try:
            day_dir = PTZ_PPE_KEYFRAME_DIR / time.strftime("%Y%m%d", time.localtime(v.ts))
            day_dir.mkdir(parents=True, exist_ok=True)
            reason = "_".join(v.types) or "ppe"
            path = day_dir / f"ptz_{reason}_{int(v.ts)}.jpg"
            path.write_bytes(image_bytes)
            v.keyframe_path = str(path)
        except Exception:
            logger.exception("save PTZ PPE keyframe failed")

    def _publish_mqtt_alert(self, v: _Violation) -> None:
        if self._ptz is None or not self._ptz.connected:
            return
        try:
            import json as _json
            payload = {
                "ts": v.ts,
                "event_type": v.event_type,
                "types": v.types,
                "lab_coat_conf": round(v.lab_coat_conf, 3),
                "gloves_conf": round(v.gloves_conf, 3),
                "keyframe_path": v.keyframe_path,
                "source": "ptz_tracker",
            }
            self._ptz._client.publish("lab/ppe_violation", _json.dumps(payload))
            logger.warning(
                "PPE event: %s %s (lab_coat=%.2f gloves=%.2f)",
                v.event_type,
                ",".join(v.types),
                v.lab_coat_conf,
                v.gloves_conf,
            )
        except Exception:
            logger.exception("publish MQTT alert failed")

    def _publish_feishu_alert(self, v: _Violation, image_bytes: bytes | None = None) -> None:
        if not image_bytes:
            return
        if "no_gloves" not in v.types:
            return
        if not _ptz_feishu_alert_enabled():
            logger.info("Feishu no-gloves alert skipped: PTZ_FEISHU_ALERT_ENABLED=0")
            return
        try:
            from backend.feishu_notifier import FeishuApiError, FeishuConfigError, FeishuNotifier

            camera_id = self._cfg.video.wvd_camera or f"opencv:{self._cfg.video.camera_index}"
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v.ts))
            text = (
                "实验室手套告警\n"
                f"摄像头: {camera_id}\n"
                f"违规项: {_format_ppe_types_zh(v.types)}\n"
                f"提示: {_build_ppe_prompt(v.types)}\n"
                f"时间: {ts}\n"
                f"手套置信度: {v.gloves_conf:.2f}"
            )
            if v.keyframe_path:
                text += f"\n关键帧: {v.keyframe_path}"
            result = FeishuNotifier.from_env().send_text_and_image(
                text=text,
                image_bytes=image_bytes,
                filename=f"ptz_no_gloves_{int(v.ts)}.jpg",
            )
            logger.info(
                "Feishu no-gloves alert sent: text=%s image=%s messages=%d",
                result.text_sent,
                result.image_sent,
                len(result.message_ids),
            )
        except FeishuConfigError as exc:
            logger.warning("Feishu alert skipped: %s", exc)
        except FeishuApiError:
            logger.exception("publish Feishu alert failed")
        except Exception:
            logger.exception("publish Feishu alert failed")

    # ---- Main loop ----
    def _run_loop(self) -> None:
        assert self._source is not None and self._detector is not None
        assert self._engine is not None and self._ptz is not None

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]
        read_failures = 0
        last_frame_ts = time.monotonic()
        last_reopen_ts = 0.0

        while not self._stop_evt.is_set():
            try:
                ok, frame = self._source.read()
            except Exception:
                logger.exception("video source read error")
                ok, frame = False, None
            if not ok or frame is None:
                read_failures += 1
                now_fail = time.monotonic()
                if (
                    (read_failures >= 60 or now_fail - last_frame_ts >= 3.0)
                    and now_fail - last_reopen_ts >= 2.0
                ):
                    last_reopen_ts = now_fail
                    with self._lock:
                        self._shared.error = (
                            f"video source read failed {read_failures} times; reopening source"
                        )
                    logger.warning(
                        "PTZ video source read stalled; reopening source mode=%s camera_index=%s wvd_camera=%s failures=%d",
                        self._cfg.video.mode,
                        self._cfg.video.camera_index,
                        self._cfg.video.wvd_camera,
                        read_failures,
                    )
                    try:
                        self._source.release()
                    except Exception:
                        logger.exception("video source release during reconnect failed")
                    try:
                        if self._source.open():
                            read_failures = 0
                        else:
                            logger.warning("PTZ video source reopen returned false")
                    except Exception:
                        logger.exception("video source reopen failed")
                time.sleep(0.02)
                continue
            read_failures = 0
            last_frame_ts = time.monotonic()
            with self._lock:
                if self._shared.error:
                    self._shared.error = None

            frame_h, frame_w = frame.shape[:2]

            try:
                persons = self._detector.detect_persons(frame)
            except Exception:
                logger.exception("person detection error")
                persons = []

            now = time.monotonic()
            new_compliance_this_frame = False
            if now - self._last_compliance_time >= self._cfg.tracking.compliance_interval:
                try:
                    self._last_compliance = self._detector.check_compliance(frame)
                    new_compliance_this_frame = True
                except Exception:
                    logger.exception("compliance detection error")
                self._last_compliance_time = now

            # 杩濊鍛婅锛氬彧鍦ㄦ湁浜?+ 鏈夋柊鐨勫悎瑙勬娴嬬粨鏋滄椂鏇存柊璁℃暟
            pending_alerts: list[_Violation] = []
            if new_compliance_this_frame and self._has_actor_signal(persons, self._last_compliance):
                pending_alerts = self._update_alert_state(self._last_compliance, time.time())

            # 璺熻釜鎺у埗
            if self._engine.state in (TrackState.TRACKING, TrackState.LOST):
                try:
                    if (
                        self._cfg.tracking.require_remote_mqtt_client
                        and now - self._last_remote_client_check_ts >= 1.0
                    ):
                        self._last_remote_client_check_ts = now
                        if not _get_mqtt_remote_clients(self._cfg.mqtt.port):
                            logger.warning("PTZ tracking stopped: remote MQTT client disconnected")
                            self._engine.stop_tracking()
                            self._ptz.stop()
                            continue
                    d_yaw, d_pitch = self._engine.update(persons, frame_w, frame_h)
                    if self._engine.state == TrackState.TRACKING and (
                        abs(d_yaw) > 0 or abs(d_pitch) > 0
                    ):
                        self._ptz.move_relative(d_pitch, d_yaw)
                except Exception:
                    logger.exception("tracking update error")

            # FPS
            self._frame_count += 1
            elapsed = now - self._fps_time
            fps_display = self._shared.fps
            if elapsed >= 1.0:
                fps_display = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_time = now

            # 鐢婚潰鍙犲姞
            annotated = _draw_overlay(
                frame,
                persons,
                self._engine.target,
                self._last_compliance,
                self._engine.state.value,
                self._ptz.current_pitch,
                self._ptz.current_yaw,
                fps_display,
                self._ptz.connected,
                alert_active=self._alert_active,
                alert_reasons=self._alert_reasons,
            )

            ok_jpg, jpeg = cv2.imencode(".jpg", annotated, encode_params)
            if not ok_jpg:
                continue
            jpeg_bytes = jpeg.tobytes()

            for event in pending_alerts:
                threading.Thread(
                    target=self._publish_alert,
                    args=(event, jpeg_bytes),
                    daemon=True,
                ).start()

            with self._lock:
                self._shared.jpeg = jpeg_bytes
                self._shared.persons = persons
                self._shared.compliance = self._last_compliance
                self._shared.state = self._engine.state.value
                self._shared.ptz_pitch = self._ptz.current_pitch
                self._shared.ptz_yaw = self._ptz.current_yaw
                self._shared.ptz_speed = self._ptz.speed
                self._shared.mqtt_connected = self._ptz.connected
                self._shared.fps = fps_display
                self._shared.timestamp = time.time()
                self._shared.alert_active = self._alert_active
                self._shared.alert_reasons = list(self._alert_reasons)
                self._shared.violations = [v.to_dict() for v in self._violations[-20:]]

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._shared.jpeg

    def latest_snapshot(self) -> tuple[Optional[bytes], float]:
        with self._lock:
            return self._shared.jpeg, self._shared.timestamp

    def video_source_id(self) -> str:
        mode = (self._cfg.video.mode or "opencv").strip().lower()
        if mode == "opencv":
            return f"opencv:{self._cfg.video.camera_index}"
        if mode in {"camera_streaming", "cameras", "shared"}:
            return f"camera_streaming:{self._cfg.video.wvd_camera or 'unset'}"
        if mode == "wvd":
            return f"wvd:{self._cfg.video.wvd_camera or 'unset'}"
        return mode


_service = PtzTrackerService()


def shutdown_ptz_service() -> None:
    _service.shutdown()


# ---- 璺敱 ----
class SpeedRequest(BaseModel):
    speed: float


class DirectionRequest(BaseModel):
    direction: str  # up/down/left/right


@router.get("/status")
def get_status():
    return _service.status()


@router.post("/start")
def start():
    try:
        _service.ensure_started()
    except Exception as e:
        raise HTTPException(500, f"Failed to start: {e}")
    return _service.status()


@router.post("/shutdown")
def shutdown():
    _service.shutdown()
    return {"ok": True}


@router.post("/track/start")
def track_start():
    try:
        _service.start_tracking()
    except Exception as e:
        raise HTTPException(500, str(e))
    return _service.status()


@router.post("/track/stop")
def track_stop():
    _service.stop_tracking()
    return _service.status()


@router.post("/ptz/center")
def ptz_center():
    _service.center()
    return _service.status()


@router.post("/ptz/manual")
def ptz_manual():
    _service.enter_manual()
    return _service.status()


@router.post("/ptz/move")
def ptz_move(req: DirectionRequest):
    try:
        _service.move(req.direction)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    return _service.status()


@router.post("/alerts/clear")
def alerts_clear():
    _service._violations.clear()
    _service._alert_active = False
    _service._alert_reasons = []
    _service._violation_streak = 0
    _service._alert_clear_frames = 0
    _service._last_alert_emit_by_signature.clear()
    return _service.status()


@router.post("/ptz/speed")
def ptz_speed(req: SpeedRequest):
    try:
        _service.set_speed(req.speed)
    except Exception as e:
        raise HTTPException(500, str(e))
    return _service.status()


@router.get("/snapshot")
def snapshot(
    auto_start: bool = Query(default=False),
    timeout_ms: int = Query(default=1000, ge=0, le=5000),
):
    """Return one JPEG frame from the running PTZ tracker service."""
    if not _service.started:
        if not auto_start:
            raise HTTPException(
                503,
                "PTZ tracker service is stopped. Call POST /api/v1/ptz-tracker/start first or pass auto_start=true.",
            )
        try:
            _service.ensure_started()
        except Exception as e:
            raise HTTPException(500, f"Failed to start PTZ tracker: {e}")

    deadline = time.monotonic() + timeout_ms / 1000.0
    frame_bytes, frame_ts = _service.latest_snapshot()
    while frame_bytes is None and time.monotonic() < deadline:
        time.sleep(0.05)
        frame_bytes, frame_ts = _service.latest_snapshot()

    if frame_bytes is None:
        raise HTTPException(503, "PTZ frame is not ready")

    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-PTZ-Frame-Timestamp": str(frame_ts),
            "X-PTZ-Video-Source": _service.video_source_id(),
        },
    )


def _mjpeg_generator():
    # Keep stream requests passive so opening the page does not start MQTT/PTZ.
    if not _service.started:
        msg = b"ptz service is stopped. start the service first."
        yield (b"--frame\r\nContent-Type: text/plain\r\n\r\n" + msg + b"\r\n")
        return

    target_interval = 1.0 / 20.0
    last_sent_ts = 0.0
    while True:
        t0 = time.monotonic()
        frame_bytes = _service.latest_jpeg()
        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n"
                b"\r\n" + frame_bytes + b"\r\n"
            )
            last_sent_ts = time.monotonic()
        elif time.monotonic() - last_sent_ts > 3.0:
            yield (
                b"--frame\r\n"
                b"Content-Type: text/plain\r\n"
                b"\r\n" + b"waiting for frame..." + b"\r\n"
            )
            last_sent_ts = time.monotonic()

        elapsed = time.monotonic() - t0
        sleep_time = target_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


@router.get("/stream")
def stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


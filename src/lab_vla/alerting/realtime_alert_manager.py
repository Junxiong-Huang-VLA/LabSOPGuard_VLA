"""
Real-Time Alert Layer for LabSOPGuard.

Multi-channel alerting:
1. WebSocket → Frontend Dashboard (real-time push)
2. MQTT → Edge devices / Mobile clients
3. Webhook → LIMS / 企业微信 / 钉钉

Alert routing by severity level, rate limiting, and alert deduplication.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import urllib.request
import urllib.error


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertChannel(str, Enum):
    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    WEBHOOK_LIMS = "webhook_lims"
    WEBHOOK_WECOM = "webhook_wecom"    # 企业微信
    WEBHOOK_DINGTALK = "webhook_dingtalk"  # 钉钉


@dataclass
class Alert:
    """An alert to be dispatched to channels."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    task_id: str = ""
    timestamp: float = 0.0
    source: str = "lab_sop_guard"
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if not self.alert_id:
            raw = f"{self.alert_type}_{self.task_id}_{self.timestamp}"
            self.alert_id = hashlib.md5(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Alert Rate Limiter & Deduplicator
# ---------------------------------------------------------------------------

class AlertThrottler:
    """Prevents alert flooding with rate limiting and deduplication."""

    def __init__(self, cooldown_sec: float = 60.0, max_per_minute: int = 20):
        self.cooldown_sec = cooldown_sec
        self.max_per_minute = max_per_minute
        self._last_sent: Dict[str, float] = {}
        self._recent_count: List[float] = []

    def should_send(self, alert: Alert) -> bool:
        """Check if alert should be sent (not throttled)."""
        key = f"{alert.alert_type}_{alert.severity.value}"
        now = time.time()

        # Cooldown check
        last = self._last_sent.get(key, 0)
        if now - last < self.cooldown_sec:
            return False

        # Rate limit check
        self._recent_count = [t for t in self._recent_count if now - t < 60.0]
        if len(self._recent_count) >= self.max_per_minute:
            return False

        self._last_sent[key] = now
        self._recent_count.append(now)
        return True


# ---------------------------------------------------------------------------
# Channel 1: WebSocket Server
# ---------------------------------------------------------------------------

class WebSocketAlertServer:
    """WebSocket server for pushing alerts to frontend dashboard.

    Uses a simple threading-based approach. For production, replace with
    asyncio + websockets or Flask-SocketIO.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5002):
        self.host = host
        self.port = port
        self._clients: List[Any] = []
        self._alert_buffer: List[Dict[str, Any]] = []
        self._max_buffer = 200
        self._running = False
        self._server = None

    def start(self) -> None:
        """Start WebSocket server in background thread."""
        self._running = True
        try:
            import asyncio
            import websockets

            async def handler(websocket):
                self._clients.append(websocket)
                try:
                    # Send recent alerts on connect
                    for alert in self._alert_buffer[-20:]:
                        await websocket.send(json.dumps(alert))
                    # Keep connection alive
                    async for _ in websocket:
                        pass
                finally:
                    self._clients.remove(websocket)

            async def main():
                async with websockets.serve(handler, self.host, self.port):
                    await asyncio.Future()  # Run forever

            def run_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(main())

            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()

        except ImportError:
            # websockets not installed, use simple HTTP SSE fallback
            self._start_sse_fallback()

    def _start_sse_fallback(self) -> None:
        """Start HTTP SSE fallback when websockets is unavailable."""
        try:
            from flask import Flask, Response
            import werkzeug.serving

            app = Flask(__name__)

            @app.route("/ws/alerts")
            def alert_stream():
                def generate():
                    last_idx = len(self._alert_buffer)
                    while True:
                        if len(self._alert_buffer) > last_idx:
                            for alert in self._alert_buffer[last_idx:]:
                                yield f"data: {json.dumps(alert)}\n\n"
                            last_idx = len(self._alert_buffer)
                        time.sleep(1.0)

                return Response(generate(), mimetype="text/event-stream")

            thread = threading.Thread(
                target=lambda: app.run(host=self.host, port=self.port, threaded=True),
                daemon=True,
            )
            thread.start()
        except Exception:
            pass

    def push_alert(self, alert: Alert) -> None:
        """Push alert to all connected clients."""
        alert_dict = alert.to_dict()
        self._alert_buffer.append(alert_dict)
        if len(self._alert_buffer) > self._max_buffer:
            self._alert_buffer = self._alert_buffer[-self._max_buffer:]

        # Try to send to connected clients
        msg = json.dumps(alert_dict, ensure_ascii=False)
        dead = []
        for client in self._clients:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(client.send(msg))
            except Exception:
                dead.append(client)

        for d in dead:
            self._clients.remove(d)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Channel 2: MQTT Client
# ---------------------------------------------------------------------------

class MQTTAlertClient:
    """MQTT client for pushing alerts to edge devices / mobile."""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic_prefix: str = "labsopguard/alerts",
        username: str = "",
        password: str = "",
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_prefix = topic_prefix
        self.username = username
        self.password = password
        self._client = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client()

            if self.username:
                self._client.username_pw_set(self.username, self.password)

            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

            self._client.connect(self.broker_host, self.broker_port, keepalive=60)
            self._client.loop_start()
            return True

        except Exception:
            self._connected = False
            return False

    def _on_connect(self, client, userdata, flags, rc):
        self._connected = rc == 0

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False

    def publish_alert(self, alert: Alert) -> bool:
        """Publish alert to MQTT topic."""
        if not self._client or not self._connected:
            return False

        try:
            topic = f"{self.topic_prefix}/{alert.severity.value}/{alert.alert_type}"
            payload = alert.to_json()
            self._client.publish(topic, payload, qos=1)
            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Channel 3: Webhook Integrations
# ---------------------------------------------------------------------------

class WebhookDispatcher:
    """Dispatches alerts via webhooks to LIMS, 企业微信, 钉钉."""

    def __init__(self, config: Dict[str, Any]):
        self.lims_url = config.get("lims_url", "")
        self.wecom_url = config.get("wecom_url", "")  # 企业微信 webhook URL
        self.dingtalk_url = config.get("dingtalk_url", "")  # 钉钉 webhook URL
        self.dingtalk_secret = config.get("dingtalk_secret", "")
        self._throttle = AlertThrottler(cooldown_sec=120.0)

    def dispatch(self, alert: Alert) -> Dict[str, bool]:
        """Dispatch alert to all configured webhook endpoints.

        Returns dict of channel -> success.
        """
        results = {}

        if not self._throttle.should_send(alert):
            return {ch: False for ch in ["lims", "wecom", "dingtalk"]}

        # LIMS
        if self.lims_url:
            results["lims"] = self._send_lims(alert)

        # 企业微信
        if self.wecom_url:
            results["wecom"] = self._send_wecom(alert)

        # 钉钉
        if self.dingtalk_url:
            results["dingtalk"] = self._send_dingtalk(alert)

        return results

    def _send_lims(self, alert: Alert) -> bool:
        """Send to LIMS system."""
        try:
            payload = {
                "alert_id": alert.alert_id,
                "type": alert.alert_type,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "task_id": alert.task_id,
                "timestamp": alert.timestamp,
                "source": alert.source,
                "metadata": alert.metadata,
            }
            return self._http_post(self.lims_url, payload)
        except Exception:
            return False

    def _send_wecom(self, alert: Alert) -> bool:
        """Send to 企业微信 (WeCom) group bot.

        WeCom webhook format: markdown card message.
        """
        try:
            sev_color = {
                "critical": "warning",
                "high": "warning",
                "medium": "info",
                "low": "info",
            }.get(alert.severity.value, "info")

            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "content": (
                        f'<font color="{sev_color}">[{alert.severity.value.upper()}]</font> '
                        f'**{alert.title}**\n'
                        f'> {alert.message}\n'
                        f'任务: {alert.task_id or "N/A"}\n'
                        f'时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert.timestamp))}'
                    )
                }
            }
            return self._http_post(self.wecom_url, payload)
        except Exception:
            return False

    def _send_dingtalk(self, alert: Alert) -> bool:
        """Send to 钉钉 (DingTalk) group bot.

        DingTalk webhook format: markdown message with optional sign.
        """
        try:
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "text": (
                        f"### [{alert.severity.value.upper()}] {alert.title}\n"
                        f"> {alert.message}\n\n"
                        f"**任务ID**: {alert.task_id or 'N/A'}\n"
                        f"**时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}\n"
                        f"**来源**: {alert.source}"
                    )
                }
            }

            # Add signature if secret configured
            url = self.dingtalk_url
            if self.dingtalk_secret:
                import hmac
                import base64
                timestamp = str(round(time.time() * 1000))
                string_to_sign = f"{timestamp}\n{self.dingtalk_secret}"
                hmac_code = hmac.new(
                    self.dingtalk_secret.encode("utf-8"),
                    string_to_sign.encode("utf-8"),
                    digestmod=hmac.sha256,
                ).digest()
                sign = base64.b64encode(hmac_code).decode("utf-8")
                url = f"{self.dingtalk_url}&timestamp={timestamp}&sign={sign}"

            return self._http_post(url, payload)
        except Exception:
            return False

    @staticmethod
    def _http_post(url: str, payload: Dict[str, Any], timeout: float = 5.0) -> bool:
        """Make HTTP POST request."""
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status < 400
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Unified Alert Manager
# ---------------------------------------------------------------------------

class AlertManager:
    """Unified alert manager that dispatches to all channels.

    Usage:
        manager = AlertManager(config)
        manager.start()
        manager.send_alert(Alert(...))
        manager.stop()
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._throttle = AlertThrottler(
            cooldown_sec=float(config.get("cooldown_sec", 60)),
            max_per_minute=int(config.get("max_per_minute", 20)),
        )

        # Severity routing: which channels get which severity levels
        self._routing: Dict[AlertSeverity, List[AlertChannel]] = {
            AlertSeverity.CRITICAL: [AlertChannel.WEBSOCKET, AlertChannel.MQTT,
                                     AlertChannel.WEBHOOK_LIMS, AlertChannel.WEBHOOK_WECOM,
                                     AlertChannel.WEBHOOK_DINGTALK],
            AlertSeverity.HIGH: [AlertChannel.WEBSOCKET, AlertChannel.MQTT,
                                 AlertChannel.WEBHOOK_WECOM],
            AlertSeverity.MEDIUM: [AlertChannel.WEBSOCKET, AlertChannel.MQTT],
            AlertSeverity.LOW: [AlertChannel.WEBSOCKET],
            AlertSeverity.INFO: [AlertChannel.WEBSOCKET],
        }

        # Initialize channels
        ws_cfg = config.get("websocket", {})
        self._ws_server = WebSocketAlertServer(
            host=ws_cfg.get("host", "0.0.0.0"),
            port=int(ws_cfg.get("port", 5002)),
        )

        mqtt_cfg = config.get("mqtt", {})
        self._mqtt = MQTTAlertClient(
            broker_host=mqtt_cfg.get("host", "localhost"),
            broker_port=int(mqtt_cfg.get("port", 1883)),
            topic_prefix=mqtt_cfg.get("topic_prefix", "labsopguard/alerts"),
            username=mqtt_cfg.get("username", ""),
            password=mqtt_cfg.get("password", ""),
        )

        self._webhook = WebhookDispatcher(config.get("webhooks", {}))

        self._alert_log: List[Dict[str, Any]] = []
        self._max_log = 1000

    def start(self) -> None:
        """Start all alert channels."""
        self._ws_server.start()

        mqtt_enabled = self.config.get("mqtt", {}).get("enabled", False)
        if mqtt_enabled:
            self._mqtt.connect()

    def stop(self) -> None:
        """Stop all alert channels."""
        self._ws_server.stop()
        self._mqtt.disconnect()

    def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """Send alert to appropriate channels based on severity routing."""
        if not self._throttle.should_send(alert):
            return {}

        results: Dict[str, bool] = {}
        channels = self._routing.get(alert.severity, [AlertChannel.WEBSOCKET])

        for channel in channels:
            if channel == AlertChannel.WEBSOCKET:
                self._ws_server.push_alert(alert)
                results["websocket"] = True

            elif channel == AlertChannel.MQTT:
                results["mqtt"] = self._mqtt.publish_alert(alert)

            elif channel in (AlertChannel.WEBHOOK_LIMS, AlertChannel.WEBHOOK_WECOM, AlertChannel.WEBHOOK_DINGTALK):
                webhook_results = self._webhook.dispatch(alert)
                results.update(webhook_results)

        # Log alert
        log_entry = alert.to_dict()
        log_entry["dispatch_results"] = results
        self._alert_log.append(log_entry)
        if len(self._alert_log) > self._max_log:
            self._alert_log = self._alert_log[-self._max_log:]

        return results

    def get_alert_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert log."""
        return self._alert_log[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total = len(self._alert_log)
        by_severity = {}
        by_type = {}
        for entry in self._alert_log:
            sev = entry.get("severity", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            typ = entry.get("alert_type", "unknown")
            by_type[typ] = by_type.get(typ, 0) + 1

        return {
            "total_alerts": total,
            "by_severity": by_severity,
            "by_type": by_type,
            "last_alert_time": self._alert_log[-1]["timestamp"] if self._alert_log else 0,
        }


# ---------------------------------------------------------------------------
# Convenience: Create alerts from PREGO anomalies
# ---------------------------------------------------------------------------

def alert_from_prego_anomaly(
    anomaly: Any,  # prego_tracker.Anomaly
    task_id: str = "",
) -> Alert:
    """Convert a PREGO anomaly to an Alert."""
    severity_map = {
        "critical": AlertSeverity.CRITICAL,
        "high": AlertSeverity.HIGH,
        "medium": AlertSeverity.MEDIUM,
        "low": AlertSeverity.LOW,
    }

    return Alert(
        alert_id="",
        alert_type=anomaly.anomaly_type.value if hasattr(anomaly, "anomaly_type") else str(anomaly.get("type", "unknown")),
        severity=severity_map.get(
            anomaly.severity.value if hasattr(anomaly, "severity") else str(anomaly.get("severity", "medium")),
            AlertSeverity.MEDIUM,
        ),
        title=anomaly.description_en if hasattr(anomaly, "description_en") else str(anomaly.get("description_en", "")),
        message=anomaly.description_zh if hasattr(anomaly, "description_zh") else str(anomaly.get("description_zh", "")),
        task_id=task_id,
        timestamp=anomaly.timestamp_sec if hasattr(anomaly, "timestamp_sec") else float(anomaly.get("timestamp", 0)),
        evidence=anomaly.evidence if hasattr(anomaly, "evidence") else {},
    )

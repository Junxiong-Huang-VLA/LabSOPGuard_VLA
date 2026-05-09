from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IntegratedSettings:
    host: str = os.getenv("INTEGRATED_HOST", "0.0.0.0")
    port: int = int(os.getenv("INTEGRATED_PORT", "5001"))
    debug: bool = os.getenv("INTEGRATED_DEBUG", "0") == "1"
    max_workers: int = int(os.getenv("INTEGRATED_MAX_WORKERS", "2"))

    enable_hand_detection: bool = os.getenv("INTEGRATED_ENABLE_HAND_DETECTION", "1") == "1"
    enable_ai_analysis: bool = os.getenv("INTEGRATED_ENABLE_AI_ANALYSIS", "1") == "1"
    enable_pdf: bool = os.getenv("INTEGRATED_ENABLE_PDF", "1") == "1"
    enable_step_check: bool = os.getenv("INTEGRATED_ENABLE_STEP_CHECK", "1") == "1"
    enable_video_export: bool = os.getenv("INTEGRATED_ENABLE_VIDEO_EXPORT", "1") == "1"

    keyframe_diff_threshold: float = float(os.getenv("INTEGRATED_KEYFRAME_DIFF_THRESHOLD", "18.0"))
    keyframe_min_interval_sec: float = float(os.getenv("INTEGRATED_KEYFRAME_MIN_INTERVAL_SEC", "1.2"))
    max_keyframes: int = int(os.getenv("INTEGRATED_MAX_KEYFRAMES", "12"))

    ai_model: str = os.getenv("INTEGRATED_AI_MODEL", "doubao-1.5-vision-pro-32k")
    ai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

    enable_ffmpeg: bool = os.getenv("INTEGRATED_ENABLE_FFMPEG", "0") == "1"
    api_token: str = os.getenv("INTEGRATED_API_TOKEN", "").strip()
    task_registry_filename: str = os.getenv("INTEGRATED_TASK_REGISTRY_FILE", "_task_registry.json")
    max_history_tasks: int = int(os.getenv("INTEGRATED_MAX_HISTORY_TASKS", "600"))

    monitor_long_running_sec: int = int(os.getenv("INTEGRATED_MONITOR_LONG_RUNNING_SEC", "1800"))
    monitor_failure_window_sec: int = int(os.getenv("INTEGRATED_MONITOR_FAILURE_WINDOW_SEC", "1800"))
    monitor_failure_threshold: int = int(os.getenv("INTEGRATED_MONITOR_FAILURE_THRESHOLD", "3"))
    monitor_webhook_url: str = os.getenv("INTEGRATED_MONITOR_WEBHOOK_URL", "").strip()
    monitor_webhook_min_level: str = os.getenv("INTEGRATED_MONITOR_WEBHOOK_MIN_LEVEL", "high").strip().lower()
    monitor_webhook_cooldown_sec: int = int(os.getenv("INTEGRATED_MONITOR_WEBHOOK_COOLDOWN_SEC", "900"))
    alert_state_filename: str = os.getenv("INTEGRATED_ALERT_STATE_FILE", "_alert_state.json")

    audit_log_filename: str = os.getenv("INTEGRATED_AUDIT_LOG_FILE", "_audit_log.jsonl")
    audit_max_entries: int = int(os.getenv("INTEGRATED_AUDIT_MAX_ENTRIES", "4000"))

    @property
    def app_root(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def outputs_root(self) -> Path:
        return self.app_root / "outputs"


def load_settings() -> IntegratedSettings:
    return IntegratedSettings()

"""Notification dispatchers for pipeline events."""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def send_feishu_notification(title: str, content: str, webhook_url: Optional[str] = None) -> bool:
    """Send notification via Feishu/Lark webhook."""
    url = webhook_url or os.getenv("FEISHU_WEBHOOK_URL")
    if not url:
        logger.debug("FEISHU_WEBHOOK_URL not configured, skipping notification")
        return False
    try:
        import requests
        payload = {
            "msg_type": "text",
            "content": {"text": f"[LabSOPGuard] {title}\n{content}"},
        }
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as exc:
        logger.warning("Feishu notification failed: %s", exc)
        return False


def notify_processing_complete(experiment_id: str, task_id: str = "") -> None:
    """Send notification when experiment processing completes."""
    try:
        send_feishu_notification(
            title="实验分析完成",
            content=f"实验 {experiment_id} 处理完成 (task={task_id[:8]})",
        )
    except Exception as exc:
        logger.warning("Completion notification failed: %s", exc)

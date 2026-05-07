from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .naming import DISPLAY_LABELS, display_name as rule_display_name


class DisplayNameEnhancer:
    """Interface for display-name enhancement. Stable names and IDs are never changed."""

    def enhance(self, experiment_name: str, event: Dict[str, Any], asset: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
        raise NotImplementedError


class RuleBasedDisplayNameEnhancer(DisplayNameEnhancer):
    def enhance(self, experiment_name: str, event: Dict[str, Any], asset: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
        return rule_display_name(experiment_name, event), "rule_based"


class QwenVlmDisplayNameEnhancer(DisplayNameEnhancer):
    """Enhance display names for published event materials.

    Priority order:
    1. event.display_name  — already computed by EventProposalBuilder (clean, human-readable)
    2. Qwen live call with keyframe image (when MATERIAL_DISPLAY_NAME_QWEN_ENABLED=true)
    3. rule_display_name fallback

    NOTE: evidence_summary is intentionally excluded — it contains internal
    grading metadata (grade=weak; score=...) and must not be used as a name.
    """

    # Only semantic / human-readable summary fields — NOT evidence_summary
    SUMMARY_KEYS = ("qwen_summary", "semantic_summary", "vlm_summary", "frame_summary")

    def __init__(self, *, enable_live: Optional[bool] = None, model: Optional[str] = None) -> None:
        self.enable_live = (
            str(os.getenv("MATERIAL_DISPLAY_NAME_QWEN_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}
            if enable_live is None
            else enable_live
        )
        self.model = model or os.getenv("MATERIAL_DISPLAY_NAME_QWEN_MODEL", "qwen3.6-flash")

    def enhance(self, experiment_name: str, event: Dict[str, Any], asset: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
        # 1. Use pre-computed display_name from EventProposalBuilder if clean
        existing = str(event.get("display_name") or "").strip()
        if existing and "grade=" not in existing and "score=" not in existing:
            return existing, "event_display_name"

        # 2. Semantic summaries (qwen/vlm, not evidence metadata)
        summary = self._summary(event)
        if summary:
            prefix = str(experiment_name or "experiment").strip() or "experiment"
            event_type = str(event.get("event_type") or "event")
            label = DISPLAY_LABELS.get(event_type, event_type)
            hint = self._compact(summary)
            if hint:
                return f"{prefix}-{label}-{hint}", "qwen_vlm_summary"

        # 3. Live Qwen call with keyframe image
        live = self._live_name(experiment_name, event, asset or {})
        if live:
            return live, "qwen_vlm_live"

        # 4. Rule-based fallback
        return rule_display_name(experiment_name, event), "rule_based"

    def _live_name(self, experiment_name: str, event: Dict[str, Any], asset: Dict[str, Any]) -> Optional[str]:
        if not self.enable_live:
            return None
        image = self._image_path(asset)
        if not image:
            return None
        try:
            from labsopguard.qwen_writeback import _call_frame_model

            src_name = str(event.get("source_container", {}).get("object_name") or "")
            tgt_name = str(event.get("target_container", {}).get("object_name") or "")
            objects_hint = f"{src_name} → {tgt_name}".strip(" →") if src_name or tgt_name else str(event.get("involved_objects") or "")
            prompt = (
                "Return strict JSON only: {\"display_name\":\"...\"}. "
                "Generate a concise Chinese display name (≤12 chars) for this lab event. "
                f"Experiment: {experiment_name}. Event: {event.get('event_type')}. "
                f"Objects: {objects_hint}. No IDs or timestamps."
            )
            result = _call_frame_model(self.model, image, prompt, timeout_sec=45, retries=0)
            display = (result.get("structured_result") or {}).get("display_name")
            return self._compact(str(display)) if display else None
        except Exception:
            return None

    @staticmethod
    def _image_path(asset: Dict[str, Any]) -> Optional[Path]:
        candidates = [asset.get("preview_path"), *((asset.get("keyframe_paths") or [])[:1])]
        for value in candidates:
            if value and Path(value).exists():
                return Path(value)
        return None

    @classmethod
    def _summary(cls, event: Dict[str, Any]) -> Optional[str]:
        for key in cls.SUMMARY_KEYS:
            value = event.get(key)
            if value:
                return str(value)
        extra = event.get("extra") or {}
        if isinstance(extra, dict):
            for key in cls.SUMMARY_KEYS:
                if extra.get(key):
                    return str(extra[key])
        return None

    @staticmethod
    def _compact(value: str) -> str:
        text = re.sub(r"\s+", " ", value).strip()
        text = text.strip(" .;:,_-")
        return text[:32]

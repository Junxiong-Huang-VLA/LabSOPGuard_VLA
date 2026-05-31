from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .physical_event_types import JitterProfile


DEFAULT_SIGMA_PX = 3.0
DEFAULT_LABEL_SIGMAS = {
    "reagent_bottle": 4.0,
    "sample_bottle": 4.0,
    "sample_bottle_blue": 4.0,
    "bottle": 4.0,
    "vial": 4.0,
    "tube": 4.0,
    "tube_cap": 4.0,
    "tube-cap": 4.0,
    "pipette": 4.0,
    "pipette_tip": 5.0,
    "beaker": 3.0,
    "balance": 3.0,
    "container": 4.0,
}


def load_jitter_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        root = Path(__file__).resolve().parents[4]
        sources = [
            root / "LabSOPGuard" / "configs" / "model" / "physical_event_gate.yaml",
            root / "configs" / "physical_event_gate.yaml",
        ]
    else:
        sources = [Path(path)]
    try:
        import yaml
    except Exception:
        return {}
    for source in sources:
        if not source.exists():
            continue
        try:
            payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
            if isinstance(payload, dict):
                payload["_config_path_used"] = str(source)
                return payload
        except Exception:
            continue
    return {}


def jitter_profile_for_label(label: str, config: Mapping[str, Any] | None = None) -> JitterProfile:
    normalized = str(label or "").strip().lower().replace(" ", "_")
    cfg = dict(config or {})
    profile_cfg = cfg.get("jitter_profile") if isinstance(cfg.get("jitter_profile"), Mapping) else cfg
    labels = profile_cfg.get("labels") if isinstance(profile_cfg.get("labels"), Mapping) else {}
    default_sigma = float(profile_cfg.get("default_sigma_px", DEFAULT_SIGMA_PX) or DEFAULT_SIGMA_PX)
    if normalized in labels:
        return JitterProfile(object_label=normalized, sigma_px=float(labels[normalized]), source="calibrated", sample_count=0)
    if normalized in DEFAULT_LABEL_SIGMAS:
        return JitterProfile(object_label=normalized, sigma_px=DEFAULT_LABEL_SIGMAS[normalized], source="fallback", sample_count=0)
    return JitterProfile(object_label=normalized, sigma_px=default_sigma, source="fallback", sample_count=0)

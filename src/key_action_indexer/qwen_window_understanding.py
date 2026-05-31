"""Qwen VLM per-window segment understanding (P2, AGENTS.md §14).

This module adds two things the pipeline lacked:

  1. ``write_integration_status`` — emits ``qwen_vlm_integration_status.json``
     describing whether Qwen is actually usable. When no client/config is
     available it records ``vlm_status = missing_config`` and produces NO faked
     understanding output.

  2. ``build_window_understanding`` — assembles a per-window
     ``window_vlm_understanding.json`` (+ ``.md``) from generated evidence only
     (keyframes/keyclips/previews/sample_grid + context + SOP). It NEVER takes a
     raw long video. Output enforces the observed/inferred/uncertainty
     separation and resolves evidence_refs against real material_ids.

Design notes:
  * The repo's VLM client is dependency-injected (duck-typed ``describe_scene``
    / ``enhance_item``); there is no env-var API key. "config present" therefore
    means a caller passed a usable client AND selected a real VLM mode.
  * Existing offline result fields map cleanly: strong_facts -> observed_facts,
    weak_inferences -> inferred_steps, unresolved_questions -> uncertainties.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

WINDOW_UNDERSTANDING_SCHEMA = "window_vlm_understanding.v1"
INTEGRATION_STATUS_SCHEMA = "qwen_vlm_integration_status.v1"

SUPPORTED_QWEN_MODELS = ("qwen3.5-flash", "qwen3.5-plus")

VLM_MODE_OFFLINE = "offline_metadata"
VLM_MODE_REUSE_EXISTING = "reuse_existing_vlm"
VLM_MODE_REAL_QWEN_ASYNC = "real_qwen_async"

# vlm_status values
STATUS_READY = "ready"
STATUS_MISSING_CONFIG = "missing_config"
STATUS_OFFLINE_MODE = "offline_mode"
STATUS_UNSUPPORTED_MODEL = "unsupported_model"


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    from .report_io import write_json_report

    return write_json_report(path, payload)


@dataclass
class IntegrationStatus:
    vlm_status: str
    vlm_mode: str
    model: str | None
    has_client: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.vlm_status == STATUS_READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INTEGRATION_STATUS_SCHEMA,
            "vlm_status": self.vlm_status,
            "vlm_mode": self.vlm_mode,
            "model": self.model,
            "has_client": self.has_client,
            "supported_models": list(SUPPORTED_QWEN_MODELS),
            "reasons": self.reasons,
        }


def detect_integration_status(
    *,
    vlm_mode: str = VLM_MODE_OFFLINE,
    model: str | None = None,
    has_client: bool = False,
) -> IntegrationStatus:
    """Honestly classify whether Qwen understanding can run.

    Never raises; returns a status. ``missing_config`` is reported when a real
    mode is requested but no client is available — the caller must then NOT
    fabricate understanding output.
    """
    reasons: list[str] = []

    if vlm_mode == VLM_MODE_OFFLINE:
        reasons.append("vlm_mode is offline_metadata; no Qwen call will be made.")
        return IntegrationStatus(STATUS_OFFLINE_MODE, vlm_mode, model, has_client, reasons)

    if model is not None and model not in SUPPORTED_QWEN_MODELS:
        reasons.append(
            f"model {model!r} is not one of supported {SUPPORTED_QWEN_MODELS}."
        )
        return IntegrationStatus(STATUS_UNSUPPORTED_MODEL, vlm_mode, model, has_client, reasons)

    if vlm_mode == VLM_MODE_REAL_QWEN_ASYNC and not has_client:
        reasons.append(
            "real_qwen_async requested but no VLM client injected; cannot call Qwen."
        )
        return IntegrationStatus(STATUS_MISSING_CONFIG, vlm_mode, model, has_client, reasons)

    if has_client:
        return IntegrationStatus(STATUS_READY, vlm_mode, model, has_client, reasons)

    reasons.append("No VLM client available for the requested mode.")
    return IntegrationStatus(STATUS_MISSING_CONFIG, vlm_mode, model, has_client, reasons)


def write_integration_status(
    output_dir: Path,
    *,
    vlm_mode: str = VLM_MODE_OFFLINE,
    model: str | None = None,
    has_client: bool = False,
) -> IntegrationStatus:
    status = detect_integration_status(vlm_mode=vlm_mode, model=model, has_client=has_client)
    _write_json(Path(output_dir) / "qwen_vlm_integration_status.json", status.to_dict())
    return status


# --------------------------------------------------------------------------
# Per-window understanding (§14)
# --------------------------------------------------------------------------


def assemble_window_vlm_input(
    *,
    window: Mapping[str, Any],
    materials: Sequence[Mapping[str, Any]],
    experiment_context: str = "",
    sop_text: str = "",
) -> dict[str, Any]:
    """Build VLM input for ONE window from generated evidence only.

    Never references a raw long video. Inputs are previews / sample_grid /
    per-material keyframes+keyclips + context + SOP + evidence refs.
    """
    window_id = window.get("experiment_window_id") or window.get("window_id")
    evidence_refs: list[dict[str, Any]] = []
    source_keyframes: list[str] = []
    source_keyclips: list[str] = []
    source_timestamps: list[Any] = []
    for m in materials:
        evidence_refs.append(
            {
                "material_id": m.get("material_id"),
                "evidence_bundle_id": m.get("evidence_bundle_id"),
                "action_type": m.get("action_type"),
            }
        )
        for k in ("first_keyframe", "third_keyframe"):
            if m.get(k):
                source_keyframes.append(str(m[k]))
        for k in ("first_keyclip", "third_keyclip", "side_by_side_keyclip"):
            if m.get(k):
                source_keyclips.append(str(m[k]))
        if m.get("peak_global_timestamp_us") is not None:
            source_timestamps.append(m["peak_global_timestamp_us"])

    return {
        "window_id": window_id,
        "third_view_realtime_preview": window.get("third_view_realtime_preview"),
        "first_view_realtime_preview": window.get("first_view_realtime_preview"),
        "side_by_side_realtime_preview": window.get("side_by_side_realtime_preview"),
        "sample_grid": window.get("sample_grid"),
        "experiment_context": experiment_context,
        "sop_text": sop_text,
        "evidence_refs": evidence_refs,
        "source_keyframes": source_keyframes,
        "source_keyclips": source_keyclips,
        "source_timestamps": source_timestamps,
        # explicit marker: raw long video is never an input
        "raw_long_video_used": False,
    }


def _resolve_refs(
    refs: Sequence[Mapping[str, Any]] | Sequence[str],
    known_material_ids: set[str],
) -> tuple[list[Any], list[Any]]:
    """Split evidence_refs into resolved (real material_id) vs unresolved."""
    resolved: list[Any] = []
    unresolved: list[Any] = []
    for ref in refs:
        mid = ref.get("material_id") if isinstance(ref, Mapping) else ref
        if mid and str(mid) in known_material_ids:
            resolved.append(ref)
        else:
            unresolved.append(ref)
    return resolved, unresolved


def build_window_understanding(
    *,
    window: Mapping[str, Any],
    materials: Sequence[Mapping[str, Any]],
    vlm_raw: Mapping[str, Any] | None,
    status: IntegrationStatus,
    segment_title: str = "",
    experiment_context: str = "",
    sop_text: str = "",
) -> dict[str, Any]:
    """Produce a §14 window_vlm_understanding record.

    If ``status`` is not ready OR ``vlm_raw`` is None, the record is marked with
    the status and contains NO fabricated observed_facts/inferred_steps — only
    the assembled evidence refs and an explicit "not generated" note.

    ``vlm_raw`` (when present) is the existing offline/online VLM result whose
    fields map: strong_facts->observed_facts, weak_inferences->inferred_steps,
    unresolved_questions->uncertainties.
    """
    window_id = window.get("experiment_window_id") or window.get("window_id")
    vlm_input = assemble_window_vlm_input(
        window=window, materials=materials,
        experiment_context=experiment_context, sop_text=sop_text,
    )
    known_ids = {str(m.get("material_id")) for m in materials if m.get("material_id")}

    base = {
        "schema_version": WINDOW_UNDERSTANDING_SCHEMA,
        "window_id": window_id,
        "segment_title": segment_title,
        "vlm_model": status.model,
        "vlm_status": status.vlm_status,
        "evidence_refs": vlm_input["evidence_refs"],
        "source_keyframes": vlm_input["source_keyframes"],
        "source_keyclips": vlm_input["source_keyclips"],
        "source_timestamps": vlm_input["source_timestamps"],
        "raw_long_video_used": False,
    }

    if not status.ready or not vlm_raw:
        base.update(
            {
                "observed_facts": [],
                "inferred_steps": [],
                "possible_experiment_phase": None,
                "involved_objects": [],
                "involved_instruments": [],
                "key_actions_summary": "",
                "uncertainties": [
                    "VLM understanding not generated: "
                    + (status.reasons[0] if status.reasons else status.vlm_status)
                ],
                "unsupported_claims": [],
                "understanding_generated": False,
            }
        )
        return base

    # Map existing VLM result fields with strict separation.
    observed = list(vlm_raw.get("strong_facts") or vlm_raw.get("observed_facts") or [])
    inferred = list(vlm_raw.get("weak_inferences") or vlm_raw.get("inferred_steps") or [])
    uncertainties = list(
        vlm_raw.get("unresolved_questions") or vlm_raw.get("uncertainties") or []
    )
    claimed_refs = vlm_raw.get("evidence_refs") or vlm_input["evidence_refs"]
    _, unresolved = _resolve_refs(claimed_refs, known_ids)
    unsupported = list(vlm_raw.get("unsupported_claims") or [])
    # Any ref the VLM cited that does not resolve is flagged, not trusted.
    for ref in unresolved:
        unsupported.append({"unresolved_evidence_ref": ref})

    base.update(
        {
            "observed_facts": observed,
            "inferred_steps": inferred,
            "possible_experiment_phase": vlm_raw.get("possible_experiment_phase"),
            "involved_objects": list(vlm_raw.get("visible_objects")
                                     or vlm_raw.get("involved_objects") or []),
            "involved_instruments": list(vlm_raw.get("involved_instruments") or []),
            "key_actions_summary": vlm_raw.get("visual_scene_summary")
            or vlm_raw.get("key_actions_summary") or "",
            "uncertainties": uncertainties,
            "unsupported_claims": unsupported,
            "understanding_generated": True,
        }
    )
    return base


def render_understanding_markdown(record: Mapping[str, Any]) -> str:
    lines = [
        f"# 实验片段理解 — {record.get('segment_title') or record.get('window_id')}",
        "",
        f"- window_id: {record.get('window_id')}",
        f"- vlm_status: {record.get('vlm_status')}",
        f"- vlm_model: {record.get('vlm_model')}",
        f"- 原始长视频是否被使用: {record.get('raw_long_video_used')}",
        "",
        "## 可见事实 (observed_facts)",
    ]
    for f in record.get("observed_facts") or ["(无)"]:
        lines.append(f"- {f}")
    lines += ["", "## 推测步骤 (inferred_steps，标注为推测)"]
    for f in record.get("inferred_steps") or ["(无)"]:
        lines.append(f"- {f}")
    lines += ["", "## 不确定性 (uncertainties)"]
    for f in record.get("uncertainties") or ["(无)"]:
        lines.append(f"- {f}")
    lines += ["", "## 不支持的判断 (unsupported_claims)"]
    for f in record.get("unsupported_claims") or ["(无)"]:
        lines.append(f"- {f}")
    return "\n".join(lines) + "\n"


def write_window_understanding(
    window_dir: Path, record: Mapping[str, Any]
) -> dict[str, str]:
    out_json = _write_json(Path(window_dir) / "window_vlm_understanding.json", record)
    md = render_understanding_markdown(record)
    out_md = Path(window_dir) / "window_vlm_understanding.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    return {"json": str(out_json), "md": str(out_md)}


def build_understanding_report(
    records: Sequence[Mapping[str, Any]], status: IntegrationStatus
) -> dict[str, Any]:
    generated = [r for r in records if r.get("understanding_generated")]
    unsupported_total = sum(len(r.get("unsupported_claims") or []) for r in records)
    refs_total = sum(len(r.get("evidence_refs") or []) for r in records)
    return {
        "schema_version": "qwen_window_understanding_report.v1",
        "vlm_status": status.vlm_status,
        "window_count": len(records),
        "understanding_generated_count": len(generated),
        "unsupported_claims_total": unsupported_total,
        "evidence_refs_total": refs_total,
        "raw_long_video_used_anywhere": any(
            r.get("raw_long_video_used") for r in records
        ),
    }

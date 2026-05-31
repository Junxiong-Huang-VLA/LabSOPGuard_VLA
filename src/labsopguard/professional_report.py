from __future__ import annotations

import base64
import json
import ntpath
import os
import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


REPORT_SCHEMA_VERSION = "professional_experiment_report.v1"
_WEASYPRINT_DLL_DIRECTORY_HANDLES: List[Any] = []
_WEASYPRINT_DLL_DIRECTORY_PATHS: List[str] = []

STANDARD_REPORT_SECTIONS: List[Dict[str, str]] = [
    {"id": "cover", "title": "封面", "purpose": "识别报告与数据来源"},
    {"id": "executive_summary", "title": "执行摘要", "purpose": "概括结论、证据充分性和核心指标"},
    {"id": "scope", "title": "范围与数据来源", "purpose": "说明报告覆盖范围、分析模块与限制"},
    {"id": "report_reliability", "title": "报告可信度", "purpose": "标注报告归档等级、证据强度和人工复核要求"},
    {"id": "evidence_gallery", "title": "证据图谱", "purpose": "展示关键帧、标注帧和跨视角证据图"},
    {"id": "key_findings", "title": "关键结论", "purpose": "沉淀最重要的发现和证据依据"},
    {"id": "procedure_assessment", "title": "步骤执行评估", "purpose": "评估结构化 SOP/实验步骤"},
    {"id": "sop_compliance_matrix", "title": "SOP 标准对照", "purpose": "对照标准步骤、实际证据、偏差和复核建议"},
    {"id": "key_action_evidence", "title": "关键动作证据", "purpose": "记录由物理证据支撑的关键动作"},
    {"id": "multiview_evidence", "title": "双视角证据", "purpose": "说明第一人称和第三人称视角的互证关系"},
    {"id": "risk_alerts", "title": "风险与异常", "purpose": "说明风险、异常、告警和复核建议"},
    {"id": "materials_traceability", "title": "关键素材与追溯", "purpose": "建立报告结论到素材索引的追溯关系"},
    {"id": "overall_assessment", "title": "综合评估与建议", "purpose": "给出总体判断和后续操作建议"},
    {"id": "limitations_audit", "title": "局限性与审计信息", "purpose": "保留模型、版本、时间和回退信息"},
    {"id": "signature_page", "title": "签字页", "purpose": "保留系统生成标识、审核人和批准人签署栏"},
]

STANDARD_REPORT_SCHEMA: Dict[str, Any] = {
    "schema_version": REPORT_SCHEMA_VERSION,
    "cover": {
        "report_title": "string",
        "experiment_name": "string",
        "experiment_id": "string",
        "run_id": "string",
        "result_version": "string",
        "generated_at": "string",
        "qwen_model": "string",
    },
    "executive_summary": {
        "overall_conclusion": "string",
        "summary": "string",
        "evidence_sufficiency": "string",
        "key_metrics": [{"label": "string", "value": "string"}],
    },
    "scope": {
        "description": "string",
        "data_sources": ["string"],
        "analysis_modules": ["string"],
        "limitations": ["string"],
    },
    "report_reliability": {
        "grade": "string",
        "archive_decision": "string",
        "summary": "string",
        "drivers": [{"label": "string", "value": "string", "interpretation": "string"}],
    },
    "evidence_gallery": {
        "summary": "string",
        "images": [{"label": "string", "caption": "string", "image_path": "string"}],
    },
    "key_findings": [
        {
            "finding": "string",
            "evidence": "string",
            "impact": "string",
            "confidence": "string",
        }
    ],
    "procedure_assessment": {
        "summary": "string",
        "steps": [
            {
                "index": "string",
                "step_name": "string",
                "status": "string",
                "time_range": "string",
                "confidence": "string",
                "evidence_count": "string",
                "assessment": "string",
            }
        ],
    },
    "sop_compliance_matrix": {
        "summary": "string",
        "rows": [
            {
                "index": "string",
                "standard_step": "string",
                "required_evidence": "string",
                "observed_evidence": "string",
                "decision": "string",
                "score": "string",
                "deviation": "string",
                "review_action": "string",
            }
        ],
    },
    "key_action_evidence": {
        "summary": "string",
        "actions": [
            {
                "action_id": "string",
                "action_type": "string",
                "action_name": "string",
                "time_range": "string",
                "duration": "string",
                "objects_en": ["string"],
                "micro_segments": "string",
                "evidence_summary": "string",
                "review_status": "string",
                "evidence_images": [{"label": "string", "caption": "string", "image_path": "string"}],
                "micro_actions": [
                    {
                        "micro_id": "string",
                        "action_name": "string",
                        "time_range": "string",
                        "objects_en": ["string"],
                        "confidence": "string",
                        "evidence_image_path": "string",
                    }
                ],
            }
        ],
    },
    "multiview_evidence": {
        "summary": "string",
        "segments": [
            {
                "action_id": "string",
                "action_name": "string",
                "time_range": "string",
                "global_time_range": "string",
                "views": [
                    {
                        "view": "first_person | third_person",
                        "view_label": "string",
                        "role": "string",
                        "time_range": "string",
                        "clip_url": "string",
                        "annotated_clip_url": "string",
                        "source_video": "string",
                        "yolo_detection_count": "string",
                        "top_objects_en": ["string"],
                        "keyframes": [{"label": "string", "image_path": "string"}],
                    }
                ],
                "cross_view_assessment": "string",
            }
        ],
        "alignment": {
            "offset_sec": "string",
            "confidence": "string",
            "method": "string",
            "interpretation": "string",
        },
    },
    "risk_alerts": {
        "summary": "string",
        "alerts": [
            {
                "severity": "string",
                "rule": "string",
                "time_or_frame": "string",
                "evidence": "string",
                "recommendation": "string",
                "count": "string",
                "evidence_image_path": "string",
            }
        ],
    },
    "materials_traceability": {
        "summary": "string",
        "materials": [
            {
                "material_name": "string",
                "event_type": "string",
                "time_range": "string",
                "evidence_grade": "string",
                "related_objects_en": ["string"],
                "evidence_image_path": "string",
            }
        ],
    },
    "overall_assessment": {
        "assessment": "string",
        "recommendations": [
            {
                "priority": "high | medium | low",
                "recommendation": "string",
                "basis": "string",
            }
        ],
        "human_review_points": ["string"],
    },
    "limitations_audit": {
        "limitations": ["string"],
        "audit_metadata": {
            "report_schema_version": "string",
            "generated_at": "string",
            "qwen_model": "string",
            "qwen_used": "boolean",
            "analysis_model": "string",
            "result_version": "string",
            "fallback_reason": "string",
        },
    },
    "signature_page": {
        "system_generated_id": "string",
        "reviewer": "string",
        "approver": "string",
        "review_date": "string",
        "approval_date": "string",
        "statement": "string",
    },
}

ANALYSIS_MODULES = [
    "标准实验分析",
    "视频标注分析",
    "关键动作索引 Key-action indexing",
    "关键素材索引",
]

ACTION_TYPE_LABELS = {
    "complete_experiment_episode": "完整实验片段",
    "liquid_transfer": "液体转移",
    "container_state_change": "容器状态变化",
    "panel_operation": "设备面板操作",
    "weighing": "称量操作",
    "hand_object_interaction": "手部与对象交互",
    "direct_visual_episode": "直接视觉证据片段",
}


def _clean_text(value: Any, fallback: str = "-") -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text or fallback


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _format_range(start: Any, end: Any) -> str:
    if start is None and end is None:
        return "-"
    start_num = _numeric_value(start)
    end_num = _numeric_value(end)
    if start_num is not None or end_num is not None:
        start_text = f"{start_num:.1f}s" if start_num is not None else "-"
        end_text = f"{end_num:.1f}s" if end_num is not None else "-"
        return f"{start_text} - {end_text}"
    start_text = _clean_text(start, "-").replace("T", " ")
    end_text = _clean_text(end, "-").replace("T", " ")
    return f"{start_text} - {end_text}"


def _format_duration(seconds: Any) -> str:
    value = _number(seconds, 0.0)
    return f"{value:.1f}s" if value > 0 else "-"


def _clean_object_label(value: Any) -> str:
    raw = _clean_text(value, "").strip("'\" ")
    if any(char in raw for char in "[]{}") or " " in raw:
        return ""
    text = re.sub(r"[^A-Za-z0-9_+/\-]", "", raw)
    if not text or len(text) > 56:
        return ""
    return text


def _unique(values: Iterable[str], limit: int = 8) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        cleaned = _clean_text(value, "")
        key = cleaned.lower()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
        if len(result) >= limit:
            break
    return result


def _action_type(value: Any) -> str:
    return _clean_text(str(value or "").strip().lower().replace(" ", "_"), "key_action")


def _friendly_action_name(value: Any) -> str:
    raw = _action_type(value)
    return ACTION_TYPE_LABELS.get(raw, raw.replace("_", " "))


def _status_review(status: Any) -> str:
    status_text = _clean_text(status, "candidate")
    if status_text in {"confirmed", "completed", "analyzed"}:
        return "证据较充分，可进入常规复核"
    if status_text in {"inferred", "candidate"}:
        return "候选或推断结果，需要人工复核"
    if status_text in {"failed", "error"}:
        return "分析失败或证据不足，需要重新检查"
    return "需要结合原始视频复核"


def _key_action_title(segment: Dict[str, Any], index: int) -> str:
    text_description = segment.get("text_description") or {}
    action = _friendly_action_name(text_description.get("action_type") or segment.get("segment_type"))
    objects = _unique(
        label
        for label in [
            *[_clean_object_label(item) for item in (segment.get("yolo_label_counts") or {}).keys()],
            *[_clean_object_label(item) for item in _safe_list(segment.get("visual_keywords"))],
            *[_clean_object_label(item) for item in _safe_list(text_description.get("tools"))],
            *[_clean_object_label(item) for item in _safe_list(text_description.get("objects"))],
        ]
        if label
    )
    return f"{action}: {', '.join(objects[:4])}" if objects else f"{action} {index + 1}"


def _path_name(value: Any) -> str:
    text = _clean_text(value, "")
    if not text:
        return ""
    try:
        return Path(text).name
    except Exception:
        return text


def _top_object_labels(counts: Any, limit: int = 8) -> List[str]:
    if not isinstance(counts, dict):
        return []
    rows: List[Tuple[str, float]] = []
    for label, count in counts.items():
        cleaned = _clean_object_label(label)
        if not cleaned:
            continue
        rows.append((cleaned, _number(count)))
    rows.sort(key=lambda item: item[1], reverse=True)
    return [label for label, _ in rows[:limit]]


def _view_evidence_payload(view_key: str, view_data: Dict[str, Any]) -> Dict[str, Any]:
    view_label = "第一人称视角" if view_key == "first_person" else "第三人称视角"
    role = "操作者近距视角" if view_key == "first_person" else "环境/全局视角"
    return {
        "view": view_key,
        "view_label": view_label,
        "role": role,
        "time_range": _format_range(view_data.get("local_start_sec"), view_data.get("local_end_sec")),
        "clip_url": _clean_text(view_data.get("clip_url"), "-"),
        "annotated_clip_url": _clean_text(view_data.get("annotated_clip_url"), "-"),
        "source_video": _path_name(view_data.get("video_path")),
        "clip_file": _path_name(view_data.get("clip_path")),
        "annotated_clip_file": _path_name(view_data.get("annotated_clip_path")),
        "yolo_detection_count": str(int(_number(view_data.get("yolo_detection_count")))),
        "top_objects_en": _top_object_labels(view_data.get("yolo_label_counts")),
    }


def _load_json_file(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _copy_jsonable(value: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except Exception:
        return dict(value)


def _experiment_output_dir(context: Dict[str, Any]) -> Optional[Path]:
    experiment = context.get("experiment") or {}
    experiment_id = _clean_text(experiment.get("experiment_id"), "")
    if not experiment_id:
        return None
    candidate = _project_root() / "outputs" / "experiments" / experiment_id
    return candidate if candidate.exists() else None


def _path_if_exists(value: Any) -> str:
    text = _clean_text(value, "")
    if not text:
        return ""
    try:
        path = Path(text)
    except Exception:
        return ""
    return str(path) if path.exists() else ""


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    text = _clean_text(value, "")
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _session_start_datetime(experiment_dir: Path) -> Optional[datetime]:
    manifest = _load_json_file(experiment_dir / "key_action_index" / "manifest.json")
    if not isinstance(manifest, dict):
        return None
    return _parse_iso_datetime(manifest.get("session_start_time"))


def _local_seconds_from_global(value: Any, session_start: Optional[datetime]) -> Optional[float]:
    current = _parse_iso_datetime(value)
    if current is None or session_start is None:
        return None
    try:
        return max(0.0, (current - session_start).total_seconds())
    except Exception:
        return None


def _image_ref(label: str, caption: str, image_path: Any) -> Dict[str, str]:
    path = _path_if_exists(image_path)
    return {"label": label, "caption": caption, "image_path": path} if path else {}


def _nearest_frame_path(experiment_dir: Path, frame_idx: Any, *, stream: str = "stream_00") -> str:
    frame_num = int(_number(frame_idx, -1))
    if frame_num < 0:
        return ""
    frame_dir = experiment_dir / "artifacts" / "frames" / stream
    if not frame_dir.exists():
        return ""
    exact = frame_dir / f"frame_{frame_num:06d}.jpg"
    if exact.exists():
        return str(exact)
    candidates = []
    for path in frame_dir.glob("frame_*.jpg"):
        match = re.search(r"frame_(\d+)\.jpg$", path.name)
        if match:
            candidates.append((abs(int(match.group(1)) - frame_num), path))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    return str(candidates[0][1])


def _alert_recommendation(rule_id: str, title: str) -> str:
    key = f"{rule_id} {title}".lower()
    if "goggle" in key or "护目镜" in title:
        return "立即佩戴护目镜；复核相关时间段是否存在遮挡、误检或人员未进入操作区。"
    if "lab_coat" in key or "实验服" in title:
        return "补充实验服或躯干防护；复核画面中实验服是否被遮挡。"
    if "glove" in key or "手套" in title:
        return "涉及手部接触样品、容器或试剂前必须佩戴防护手套。"
    return "结合原始视频和现场记录进行人工复核。"


def _detailed_alert_rows(experiment_dir: Path, limit: int = 8) -> List[Dict[str, Any]]:
    analysis = _load_json_file(experiment_dir / "analysis" / "analysis.json")
    if not isinstance(analysis, list):
        return []
    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for frame in analysis:
        if not isinstance(frame, dict):
            continue
        for detail in _safe_list(frame.get("alert_details")):
            if not isinstance(detail, dict):
                continue
            rule_id = _clean_text(detail.get("rule_id"), "alert")
            title = _clean_text(detail.get("title"), rule_id)
            severity = _clean_text(detail.get("severity"), "-")
            key = (rule_id, title, severity)
            row = groups.setdefault(
                key,
                {
                    "rule": title,
                    "severity": severity,
                    "message": _clean_text(detail.get("message"), ""),
                    "rule_basis": _clean_text(detail.get("rule_basis"), ""),
                    "confidence_values": [],
                    "timestamps": [],
                    "frames": [],
                    "evidence_image_path": "",
                },
            )
            timestamp = _numeric_value(detail.get("timestamp_sec"))
            source_frame = detail.get("source_frame")
            if timestamp is not None:
                row["timestamps"].append(timestamp)
            if source_frame is not None:
                row["frames"].append(int(_number(source_frame, 0)))
            confidence = _numeric_value(detail.get("confidence"))
            if confidence is not None:
                row["confidence_values"].append(confidence)
            if not row["evidence_image_path"]:
                row["evidence_image_path"] = _nearest_frame_path(experiment_dir, source_frame)
    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    rows: List[Dict[str, Any]] = []
    for (rule_id, title, severity), row in groups.items():
        timestamps = row.pop("timestamps")
        frames = row.pop("frames")
        confidences = row.pop("confidence_values")
        count = max(len(timestamps), len(frames), 1)
        if timestamps:
            time_text = _format_range(min(timestamps), max(timestamps))
        elif frames:
            time_text = f"frame {min(frames)} - {max(frames)}"
        else:
            time_text = "-"
        if frames:
            time_text = f"{time_text}; frames {min(frames)}-{max(frames)}"
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        rows.append(
            {
                "severity": severity,
                "rule": title,
                "time_or_frame": time_text,
                "evidence": f"{row['message']} 规则依据：{row['rule_basis']}；平均置信度 {avg_confidence:.2f}。",
                "recommendation": _alert_recommendation(rule_id, title),
                "count": str(count),
                "evidence_image_path": row.get("evidence_image_path", ""),
            }
        )
    rows.sort(key=lambda item: (severity_order.get(item["severity"], 9), -int(_number(item.get("count"), 0))))
    return rows[:limit]


def _sop_compliance_rows(experiment_dir: Path) -> List[Dict[str, str]]:
    data = _load_json_file(experiment_dir / "step_candidates.json")
    if not isinstance(data, dict):
        return []
    candidates = {
        _clean_text(item.get("protocol_step_id"), ""): item
        for item in _safe_list(data.get("step_candidates"))
        if isinstance(item, dict)
    }
    decisions = {
        _clean_text(item.get("protocol_step_id"), ""): item
        for item in _safe_list(data.get("promotion_decisions"))
        if isinstance(item, dict)
    }
    bundles = {
        _clean_text(item.get("protocol_step_id"), ""): item
        for item in _safe_list(data.get("evidence_bundles"))
        if isinstance(item, dict)
    }
    rows: List[Dict[str, str]] = []
    for index, step in enumerate(_safe_list(data.get("protocol_graph"))[:16]):
        if not isinstance(step, dict):
            continue
        step_id = _clean_text(step.get("protocol_step_id"), "")
        candidate = candidates.get(step_id, {})
        decision = decisions.get(step_id, {})
        blocking = _safe_list(decision.get("blocking_issues"))
        blocking_text = ", ".join(
            {"predecessor_not_confirmed": "前序步骤未确认"}.get(_clean_text(item, ""), _clean_text(item, ""))
            for item in blocking
            if _clean_text(item, "")
        )
        deviation = blocking_text or (
            "候选证据，需复核是否满足 SOP 标准" if _clean_text(candidate.get("candidate_status"), "") == "candidate" else "-"
        )
        observed = _safe_list(candidate.get("matched_event_types"))
        observed_text = ", ".join(_clean_text(item, "") for item in observed if _clean_text(item, ""))
        matched_count = len(_safe_list(candidate.get("matched_event_ids")))
        grade = _clean_text(candidate.get("evidence_grade"), "-")
        if observed_text:
            observed_text = f"{observed_text}; refs={matched_count}; grade={grade}"
        else:
            observed_text = f"refs={matched_count}; grade={grade}"
        recommendation = _clean_text(decision.get("recommendation"), "manual_review_required")
        recommendation_text = {
            "manual_review_required_before_step_promotion": "人工复核后再提升步骤状态",
            "retain_for_human_review_or_additional_evidence": "保留候选并补充证据",
            "manual_review_required": "人工复核",
        }.get(recommendation, recommendation)
        rows.append(
            {
                "index": str(step.get("step_index", index)),
                "standard_step": _clean_text(step.get("protocol_step_name"), f"Step {index}"),
                "required_evidence": ", ".join(_clean_text(item, "") for item in _safe_list(step.get("required_event_types"))),
                "observed_evidence": observed_text or "-",
                "decision": _clean_text(decision.get("decision") or candidate.get("candidate_status"), "-"),
                "score": _format_number(decision.get("score") or candidate.get("candidate_score")),
                "deviation": deviation,
                "review_action": recommendation_text,
            }
        )
    return rows


def _micro_actions(experiment_dir: Path) -> List[Dict[str, Any]]:
    session_start = _session_start_datetime(experiment_dir)
    rows: List[Dict[str, Any]] = []
    micro_root = experiment_dir / "key_action_index" / "indexed_db" / "micro_segments"
    if not micro_root.exists():
        return rows
    for metadata_path in sorted(micro_root.glob("*/metadata.json")):
        data = _load_json_file(metadata_path)
        if not isinstance(data, dict):
            continue
        start = _local_seconds_from_global(data.get("global_start_time"), session_start)
        end = _local_seconds_from_global(data.get("global_end_time"), session_start)
        if start is None or end is None:
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)s", metadata_path.parent.name)
            if match:
                start, end = float(match.group(1)), float(match.group(2))
        primary_object = _clean_object_label(data.get("primary_object"))
        object_zh = _clean_text(data.get("primary_object_zh"), primary_object or "对象")
        interaction_type = _clean_text(data.get("interaction_type"), "hand_object_contact")
        rows.append(
            {
                "micro_id": _clean_text(data.get("micro_segment_id"), metadata_path.parent.name),
                "parent_segment_id": _clean_text(data.get("parent_segment_id"), "seg_000001"),
                "action_name": f"{object_zh}接触",
                "action_type": interaction_type,
                "time_range": _format_range(start, end),
                "objects_en": [item for item in ["gloved_hand", primary_object] if item],
                "confidence": _format_number(data.get("confidence")),
                "evidence_image_path": _path_if_exists(data.get("peak_keyframe")),
            }
        )
    return rows


def _segment_keyframes(experiment_dir: Path, segment_id: str) -> List[Dict[str, str]]:
    keyframe_dir = experiment_dir / "key_action_index" / "keyframes" / segment_id
    candidates = [
        ("第一人称中段", "操作者近距视角关键帧", keyframe_dir / "first_person_middle.jpg"),
        ("第三人称中段", "台面全局视角关键帧", keyframe_dir / "third_person_middle.jpg"),
        ("交互证据 1", "手部与对象交互代表帧", keyframe_dir / "interaction_001.jpg"),
        ("交互证据 2", "手部与对象交互补充帧", keyframe_dir / "interaction_002.jpg"),
    ]
    return [item for item in (_image_ref(label, caption, path) for label, caption, path in candidates) if item]


def _view_keyframes(experiment_dir: Path, segment_id: str, view: str) -> List[Dict[str, str]]:
    keyframe_dir = experiment_dir / "key_action_index" / "keyframes" / segment_id
    candidates = [
        ("开始", keyframe_dir / f"{view}_start.jpg"),
        ("中段", keyframe_dir / f"{view}_middle.jpg"),
        ("结束", keyframe_dir / f"{view}_end.jpg"),
    ]
    return [
        {"label": label, "image_path": _path_if_exists(path)}
        for label, path in candidates
        if _path_if_exists(path)
    ]


def _material_image_path(experiment_dir: Path, material_name: Any) -> str:
    name = _clean_text(material_name, "")
    if not name:
        return ""
    keyframe_dirs = [
        experiment_dir / "_material_review_queue" / "\u5173\u952e\u5e27",
        experiment_dir / "material_candidates" / "\u5173\u952e\u5e27",
    ]
    for keyframe_dir in keyframe_dirs:
        if not keyframe_dir.exists():
            continue
        exact = keyframe_dir / f"{name}.jpg"
        if exact.exists():
            return str(exact)
        prefix = re.sub(r"_\d+$", "", name)
        matches = sorted(keyframe_dir.glob(f"{prefix}*.jpg"))
        if matches:
            return str(matches[0])
    return ""


def _report_reliability(context: Dict[str, Any]) -> Dict[str, Any]:
    metrics = context.get("metrics") or {}
    steps = _safe_list(context.get("steps"))
    alerts = _safe_list(context.get("alerts"))
    quality = context.get("quality") or {}
    candidate_steps = sum(1 for item in steps if _clean_text(_as_dict(item).get("status"), "").lower() in {"candidate", "inferred"})
    avg_confidence = _number(metrics.get("avg_confidence"), 0.0)
    quality_status = _clean_text(quality.get("overall_status"), "unknown")
    quality_score = _format_number(quality.get("overall_score"))
    if candidate_steps or avg_confidence < 0.75 or quality_status == "fail":
        grade = "需复核后归档"
        decision = "不可直接用于合规终审"
    elif avg_confidence >= 0.85 and not alerts:
        grade = "可归档"
        decision = "可进入常规审核归档"
    else:
        grade = "有限可信"
        decision = "需抽样复核后归档"
    return {
        "grade": grade,
        "archive_decision": decision,
        "summary": (
            f"本报告自动证据链已形成，但候选步骤 {candidate_steps}/{len(steps)}，平均置信度 {avg_confidence:.3f}，"
            f"质量检查状态 {quality_status}，因此结论应作为专业复核草案使用。"
        ),
        "drivers": [
            {"label": "平均置信度", "value": f"{avg_confidence:.3f}", "interpretation": "低于高可靠自动审计阈值时需人工复核。"},
            {"label": "候选/推断步骤", "value": f"{candidate_steps}/{len(steps)}", "interpretation": "候选步骤比例越高，越不适合直接终审。"},
            {"label": "流程质量检查", "value": f"{quality_status} / {quality_score}", "interpretation": "来自关键动作索引质量评估。"},
            {"label": "风险告警", "value": str(metrics.get("alerts", len(alerts))), "interpretation": "告警必须回看原始视频和证据帧。"},
        ],
    }


def _enrich_context_from_files(context: Dict[str, Any]) -> Dict[str, Any]:
    enriched = _copy_jsonable(context)
    experiment_dir = _experiment_output_dir(enriched)
    if not experiment_dir:
        return enriched
    enriched["filesystem"] = {"experiment_dir": str(experiment_dir)}

    view_alignment = _load_json_file(experiment_dir / "key_action_index" / "metadata" / "view_alignment.json")
    if isinstance(view_alignment, dict):
        enriched["view_alignment"] = view_alignment

    quality = _load_json_file(experiment_dir / "key_action_index" / "metadata" / "process_quality_report.json")
    if isinstance(quality, dict):
        enriched["quality"] = {
            "overall_status": quality.get("overall_status"),
            "overall_score": quality.get("overall_score"),
            "failed_check_ids": _safe_list((quality.get("diagnostics") or {}).get("failed_check_ids")),
            "needs_review_check_ids": _safe_list((quality.get("diagnostics") or {}).get("needs_review_check_ids")),
            "top_recommendations": _safe_list((quality.get("diagnostics") or {}).get("top_recommendations"))[:5],
        }

    sop_rows = _sop_compliance_rows(experiment_dir)
    if sop_rows:
        enriched["sop_compliance"] = sop_rows

    detailed_alerts = _detailed_alert_rows(experiment_dir)
    if detailed_alerts:
        enriched["alerts_detailed"] = detailed_alerts
        enriched["alerts"] = detailed_alerts

    micro_actions = _micro_actions(experiment_dir)
    micro_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for item in micro_actions:
        micro_by_parent.setdefault(_clean_text(item.get("parent_segment_id"), ""), []).append(item)

    key_actions = [item for item in _safe_list(enriched.get("key_actions")) if isinstance(item, dict)]
    for action in key_actions:
        segment_id = _clean_text(action.get("action_id"), "seg_000001")
        action["evidence_images"] = _segment_keyframes(experiment_dir, segment_id)
        action["micro_actions"] = micro_by_parent.get(segment_id, [])
        for view in _safe_list(action.get("views")):
            if isinstance(view, dict):
                view["keyframes"] = _view_keyframes(experiment_dir, segment_id, _clean_text(view.get("view"), ""))
    enriched["key_actions"] = key_actions

    for item in _safe_list(enriched.get("materials")):
        if isinstance(item, dict):
            item["evidence_image_path"] = _material_image_path(experiment_dir, item.get("name"))

    gallery: List[Dict[str, str]] = []
    for action in key_actions[:1]:
        gallery.extend(_safe_list(action.get("evidence_images"))[:3])
        for micro in _safe_list(action.get("micro_actions"))[:2]:
            image = _image_ref(
                _clean_text(micro.get("action_name"), "微动作"),
                f"{_clean_text(micro.get('time_range'), '-')} · {_clean_text(micro.get('action_type'), '')}",
                micro.get("evidence_image_path"),
            )
            if image:
                gallery.append(image)
    for alert in detailed_alerts[:2]:
        image = _image_ref(
            _clean_text(alert.get("rule"), "风险告警"),
            f"{_clean_text(alert.get('time_or_frame'), '-')} · {_clean_text(alert.get('severity'), '-')}",
            alert.get("evidence_image_path"),
        )
        if image:
            gallery.append(image)
    for material in _safe_list(enriched.get("materials"))[:2]:
        if isinstance(material, dict):
            image = _image_ref(
                _clean_text(material.get("name"), "关键素材"),
                f"{_clean_text(material.get('event_type'), '-')} · {_clean_text(material.get('time_range'), '-')}",
                material.get("evidence_image_path"),
            )
            if image:
                gallery.append(image)
    seen_paths = set()
    deduped = []
    for item in gallery:
        path = item.get("image_path")
        if path and path not in seen_paths:
            deduped.append(item)
            seen_paths.add(path)
        if len(deduped) >= 8:
            break
    enriched["evidence_gallery"] = deduped
    enriched["report_reliability"] = _report_reliability(enriched)
    return enriched


def build_report_context(
    *,
    overview: Dict[str, Any],
    key_actions: Optional[Dict[str, Any]],
    materials: Dict[str, Any],
) -> Dict[str, Any]:
    steps_payload = overview.get("steps") or {}
    steps = (
        _safe_list(steps_payload.get("official"))
        or _safe_list(steps_payload.get("candidate"))
        or _safe_list(steps_payload.get("inferred"))
    )
    segments = _safe_list((key_actions or {}).get("segments"))
    material_items = _safe_list(materials.get("items"))[:16]
    alerts = _safe_list(overview.get("alerts"))[:16]
    summary = overview.get("summary") or {}
    run = overview.get("run") or {}
    experiment = overview.get("experiment") or {}

    key_action_rows: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments[:16]):
        text_description = segment.get("text_description") or {}
        object_labels = _unique(
            label
            for label in [
                *[_clean_object_label(item) for item in (segment.get("yolo_label_counts") or {}).keys()],
                *[_clean_object_label(item) for item in _safe_list(segment.get("visual_keywords"))],
                *[_clean_object_label(item) for item in _safe_list(text_description.get("tools"))],
                *[_clean_object_label(item) for item in _safe_list(text_description.get("objects"))],
            ]
            if label
        )
        first_person = segment.get("first_person") or {}
        third_person = segment.get("third_person") or {}
        cv_detection = segment.get("cv_detection") or {}
        local_start = (
            third_person.get("local_start_sec")
            or first_person.get("local_start_sec")
            or cv_detection.get("start_sec")
            or segment.get("start_time_sec")
        )
        local_end = (
            third_person.get("local_end_sec")
            or first_person.get("local_end_sec")
            or cv_detection.get("end_sec")
            or segment.get("end_time_sec")
        )
        global_start = segment.get("global_start_time")
        global_end = segment.get("global_end_time")
        action_type = _action_type(text_description.get("action_type") or segment.get("segment_type"))
        view_rows = [
            _view_evidence_payload(view_key, view_data)
            for view_key, view_data in [("first_person", first_person), ("third_person", third_person)]
            if isinstance(view_data, dict) and view_data
        ]
        key_action_rows.append(
            {
                "index": index + 1,
                "action_id": _clean_text(segment.get("segment_id") or segment.get("id"), f"KA-{index + 1:03d}"),
                "title": _key_action_title(segment, index),
                "action_type": action_type,
                "action_name": _friendly_action_name(action_type),
                "objects_en": object_labels[:8],
                "duration_sec": _number(segment.get("duration_sec")),
                "micro_segment_count": len(_safe_list(segment.get("micro_segments"))),
                "interaction_event_count": len(_safe_list(segment.get("interaction_events"))),
                "time_range": _format_range(local_start or global_start, local_end or global_end),
                "global_time_range": _format_range(global_start, global_end),
                "views": view_rows,
                "review_status": _status_review(segment.get("status") or text_description.get("status")),
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment": {
            "experiment_id": experiment.get("experiment_id"),
            "experiment_name": experiment.get("experiment_name") or experiment.get("name"),
            "description": experiment.get("description"),
        },
        "run": {
            "run_id": run.get("run_id"),
            "status": run.get("status"),
            "stage": run.get("stage"),
            "progress": run.get("progress"),
            "result_version": run.get("result_version"),
            "updated_at": run.get("updated_at"),
        },
        "metrics": {
            "frames": summary.get("frame_count", 0),
            "detections": summary.get("detection_count", 0),
            "alerts": summary.get("alert_count", len(alerts)),
            "steps": len(steps),
            "key_actions": len(segments),
            "materials": len(_safe_list(materials.get("items"))),
            "avg_confidence": summary.get("avg_confidence"),
            "model_name": summary.get("model_name"),
        },
        "scene_summary": overview.get("scene_summary") or {},
        "steps": [
            {
                "index": step.get("step_index", index + 1),
                "name": _clean_text(step.get("step_name"), f"Step {index + 1}"),
                "status": _clean_text(step.get("status"), "-"),
                "time_range": _format_range(step.get("start_time_sec"), step.get("end_time_sec")),
                "confidence": _format_number(step.get("confidence")),
                "evidence_count": str(len(_safe_list(step.get("evidence_refs")))),
                "assessment": _status_review(step.get("status")),
            }
            for index, step in enumerate(steps[:20])
        ],
        "key_actions": key_action_rows,
        "materials": [
            {
                "index": index + 1,
                "name": _clean_text(
                    item.get("display_name") or item.get("event_type") or item.get("event_id"),
                    f"Material {index + 1}",
                ),
                "event_type": _clean_text(item.get("event_type"), "material"),
                "time_range": _format_range(item.get("time_start"), item.get("time_end")),
                "evidence_grade": _clean_text(item.get("evidence_grade"), "-"),
                "related_objects_en": [
                    label
                    for label in (
                        _clean_object_label(obj)
                        for obj in _safe_list(item.get("objects") or item.get("object_labels"))
                    )
                    if label
                ][:8],
            }
            for index, item in enumerate(material_items)
        ],
        "alerts": [
            {
                "index": index + 1,
                "rule": _clean_text(alert.get("rule_name") or alert.get("rule_id"), f"Alert {index + 1}"),
                "severity": _clean_text(alert.get("severity"), "-"),
                "message": _clean_text(alert.get("message"), ""),
                "confidence": _format_number(alert.get("confidence")),
                "time_or_frame": _clean_text(alert.get("timestamp") or alert.get("frame_index"), "-"),
            }
            for index, alert in enumerate(alerts)
        ],
    }


def _standard_key_metrics(context: Dict[str, Any]) -> List[Dict[str, str]]:
    metrics = context.get("metrics") or {}
    return [
        {"label": "视频帧数", "value": str(metrics.get("frames", 0))},
        {"label": "目标检测数", "value": str(metrics.get("detections", 0))},
        {"label": "结构化步骤数", "value": str(metrics.get("steps", 0))},
        {"label": "关键动作数", "value": str(metrics.get("key_actions", 0))},
        {"label": "关键素材数", "value": str(metrics.get("materials", 0))},
        {"label": "告警数", "value": str(metrics.get("alerts", 0))},
        {"label": "平均置信度", "value": _format_number(metrics.get("avg_confidence"))},
    ]


def _cross_view_assessment(item: Dict[str, Any], alignment: Dict[str, Any]) -> str:
    views = [view for view in _safe_list(item.get("views")) if isinstance(view, dict)]
    by_view = {_clean_text(view.get("view"), ""): view for view in views}
    first = by_view.get("first_person") or {}
    third = by_view.get("third_person") or {}
    first_objects = set(_safe_list(first.get("top_objects_en")))
    third_objects = set(_safe_list(third.get("top_objects_en")))
    shared = sorted(first_objects & third_objects)
    first_count = int(_number(first.get("yolo_detection_count"), 0))
    third_count = int(_number(third.get("yolo_detection_count"), 0))
    offset = _number(alignment.get("offset_sec"), 0.0)
    confidence = _format_number(alignment.get("confidence"))
    details = [
        f"双视角时间偏移 {offset:.2f}s，同步置信度 {confidence}",
        f"第一人称检测数 {first_count}，第三人称检测数 {third_count}",
    ]
    if shared:
        details.append(f"共同验证对象：{', '.join(shared[:6])}")
    if third_count > first_count:
        details.append("第三人称视角提供更强的台面全局证据；第一人称视角用于确认操作者近距接触。")
    return "；".join(details) + "。"


def _report_extensions_from_context(context: Dict[str, Any]) -> Dict[str, Any]:
    key_actions = [item for item in _safe_list(context.get("key_actions")) if isinstance(item, dict)]
    materials = [item for item in _safe_list(context.get("materials")) if isinstance(item, dict)]
    alerts = [item for item in _safe_list(context.get("alerts_detailed") or context.get("alerts")) if isinstance(item, dict)]
    sop_rows = [item for item in _safe_list(context.get("sop_compliance")) if isinstance(item, dict)]
    gallery = [item for item in _safe_list(context.get("evidence_gallery")) if isinstance(item, dict)]
    alignment = _as_dict(context.get("view_alignment"))

    multiview_segments = []
    for index, item in enumerate(key_actions[:16]):
        views = _safe_list(item.get("views"))
        if not views:
            continue
        multiview_segments.append(
            {
                "action_id": _clean_text(item.get("action_id"), f"KA-{index + 1:03d}"),
                "action_name": _clean_text(item.get("action_name"), f"Key Action {index + 1}"),
                "time_range": _clean_text(item.get("time_range"), "-"),
                "global_time_range": _clean_text(item.get("global_time_range"), "-"),
                "views": views,
                "cross_view_assessment": _cross_view_assessment(item, alignment),
            }
        )
    multiview_count = sum(len(_safe_list(item.get("views"))) for item in multiview_segments)

    action_rows = []
    for index, item in enumerate(key_actions[:16]):
        action_rows.append(
            {
                "action_id": _clean_text(item.get("action_id"), f"KA-{index + 1:03d}"),
                "action_type": _clean_text(item.get("action_type"), "key_action"),
                "action_name": _clean_text(item.get("action_name"), f"Key Action {index + 1}"),
                "time_range": _clean_text(item.get("time_range"), "-"),
                "duration": _format_duration(item.get("duration_sec")),
                "objects_en": _safe_list(item.get("objects_en"))[:8],
                "micro_segments": str(len(_safe_list(item.get("micro_actions"))) or item.get("micro_segment_count", 0)),
                "evidence_summary": (
                    f"{item.get('interaction_event_count', 0)} 次手-物交互事件；"
                    f"{len(_safe_list(item.get('micro_actions')))} 个可定位微动作；证据图 {len(_safe_list(item.get('evidence_images')))} 张。"
                ),
                "review_status": _clean_text(item.get("review_status"), "需要人工复核"),
                "evidence_images": _safe_list(item.get("evidence_images"))[:4],
                "micro_actions": _safe_list(item.get("micro_actions"))[:8],
            }
        )

    material_rows = []
    for index, item in enumerate(materials[:12]):
        material_rows.append(
            {
                "material_name": _clean_text(item.get("name"), f"Material {index + 1}"),
                "event_type": _clean_text(item.get("event_type"), "material"),
                "time_range": _clean_text(item.get("time_range"), "-"),
                "evidence_grade": _clean_text(item.get("evidence_grade"), "-"),
                "related_objects_en": _safe_list(item.get("related_objects_en"))[:8],
                "evidence_image_path": _clean_text(item.get("evidence_image_path"), ""),
            }
        )

    return {
        "report_reliability": context.get("report_reliability") or _report_reliability(context),
        "evidence_gallery": {
            "summary": f"本页汇总 {len(gallery)} 张代表性证据图，覆盖关键动作、微动作、风险告警和关键素材。",
            "images": gallery[:8],
        },
        "sop_compliance_matrix": {
            "summary": (
                f"系统将识别结果与 {len(sop_rows)} 个 SOP 标准步骤进行对照。"
                "decision 为 hold_for_review 或 keep_candidate 的步骤必须进入人工复核。"
            ),
            "rows": sop_rows[:16],
        },
        "key_action_evidence": {
            "summary": (
                f"系统形成 {len(key_actions)} 个关键动作片段，并进一步展开为 "
                f"{sum(len(_safe_list(item.get('micro_actions'))) for item in key_actions)} 个微动作证据点。"
                "对象名保持英文标签，以便与 YOLO 检测、向量检索和素材索引一致。"
            ),
            "actions": action_rows,
        },
        "multiview_evidence": {
            "summary": (
                f"本次报告纳入 {len(multiview_segments)} 个关键动作片段的双视角证据，共 {multiview_count} 条视角记录。"
                "第一人称用于近距手部操作复核，第三人称用于台面全局状态和器具位置复核。"
            ),
            "segments": multiview_segments,
            "alignment": {
                "offset_sec": _format_number(alignment.get("offset_sec")),
                "confidence": _format_number(alignment.get("confidence")),
                "method": _clean_text(alignment.get("method"), "-"),
                "interpretation": _clean_text(alignment.get("interpretation"), "-"),
            },
        },
        "risk_alerts": {
            "summary": f"系统聚合 {len(alerts)} 类风险告警；每类告警保留时间范围、触发次数和代表性证据帧。",
            "alerts": alerts[:8],
        },
        "materials_traceability": {
            "summary": f"系统当前发布 {len(materials)} 条关键素材，支持从报告结论回溯到关键帧和片段。",
            "materials": material_rows,
        },
    }


def _fallback_report(context: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    metrics = context.get("metrics") or {}
    experiment = context.get("experiment") or {}
    run = context.get("run") or {}
    key_actions = context.get("key_actions") or []
    steps = context.get("steps") or []
    alerts = context.get("alerts") or []
    materials = context.get("materials") or []
    multiview_segments = [
        {
            "action_id": _clean_text(item.get("action_id"), f"KA-{index + 1:03d}"),
            "action_name": _clean_text(item.get("action_name"), f"Key Action {index + 1}"),
            "time_range": _clean_text(item.get("time_range"), "-"),
            "global_time_range": _clean_text(item.get("global_time_range"), "-"),
            "views": _safe_list(item.get("views")),
        }
        for index, item in enumerate(key_actions[:16])
        if _safe_list(item.get("views"))
    ]
    multiview_count = sum(len(_safe_list(item.get("views"))) for item in multiview_segments)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    experiment_name = _clean_text(experiment.get("experiment_name"), "实验")
    qwen_model = os.getenv("QWEN_REPORT_MODEL", "qwen3.6-max-preview")
    system_generated_id = "-".join(
        part
        for part in [
            "RL",
            _clean_text(experiment.get("experiment_id"), "").replace("-", "")[:8],
            _clean_text(run.get("run_id"), "").replace("-", "")[:8],
            datetime.now().strftime("%Y%m%d%H%M%S"),
        ]
        if part
    )

    findings = [
        {
            "finding": f"系统识别到 {len(steps)} 个结构化步骤和 {len(key_actions)} 个关键动作片段。",
            "evidence": "来自视频分析输出、关键动作索引和结构化步骤结果。",
            "impact": "可用于定位实验执行过程中的主要操作阶段和物理证据片段。",
            "confidence": "需要结合每个片段的状态和置信度进行复核。",
        },
        {
            "finding": f"当前告警数量为 {len(alerts)} 条。",
            "evidence": "来自规则检测、过程分析或异常标记结果。",
            "impact": "告警应作为人工复核入口，不应直接替代最终实验判断。",
            "confidence": "候选告警需要结合原始视频和现场记录确认。",
        },
    ]
    if materials:
        findings.append(
            {
                "finding": f"系统发布了 {len(materials)} 条关键素材索引。",
                "evidence": "来自关键素材发布与语义索引结果。",
                "impact": "支持后续检索、追溯和报告证据核查。",
                "confidence": "素材分级和时间范围需要按证据等级复核。",
            }
        )

    extensions = _report_extensions_from_context(context)
    base_report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "cover": {
            "report_title": f"{experiment_name} 实验分析专业报告",
            "experiment_name": experiment_name,
            "experiment_id": _clean_text(experiment.get("experiment_id"), "-"),
            "run_id": _clean_text(run.get("run_id"), "-"),
            "result_version": _clean_text(run.get("result_version"), "-"),
            "generated_at": generated_at,
            "qwen_model": qwen_model,
        },
        "executive_summary": {
            "overall_conclusion": "本报告基于 RealityLoop 自动分析结果形成，当前结论应作为实验复核与证据追溯的专业草案。",
            "summary": (
                f"系统处理 {metrics.get('frames', 0)} 帧视频数据，识别 {metrics.get('detections', 0)} 次目标检测，"
                f"形成 {len(steps)} 个结构化步骤、{len(key_actions)} 个关键动作片段、{len(materials)} 条关键素材和 {len(alerts)} 条告警。"
                "报告重点关注物理证据、手部与对象交互、微片段和可追溯素材索引。候选或推断结果已作为复核项保留。"
            ),
            "evidence_sufficiency": "证据充分性取决于视频质量、检测稳定性、关键动作片段完整性和人工复核结果。",
            "key_metrics": _standard_key_metrics(context),
        },
        "scope": {
            "description": _clean_text(experiment.get("description"), "本报告覆盖该实验运行的自动视频分析结果。"),
            "data_sources": ["实验上传视频", "结构化步骤结果", "YOLO 检测与关键动作片段", "关键素材索引"],
            "analysis_modules": ANALYSIS_MODULES,
            "limitations": ["当视频视角遮挡、动作不完整或置信度较低时，相关结论需要人工复核。"],
        },
        "report_reliability": extensions["report_reliability"],
        "evidence_gallery": extensions["evidence_gallery"],
        "key_findings": findings,
        "procedure_assessment": {
            "summary": f"系统当前形成 {len(steps)} 个结构化步骤。candidate 或 inferred 状态表示仍需人工确认。",
            "steps": [
                {
                    "index": str(item.get("index", index + 1)),
                    "step_name": _clean_text(item.get("name"), f"Step {index + 1}"),
                    "status": _clean_text(item.get("status"), "-"),
                    "time_range": _clean_text(item.get("time_range"), "-"),
                    "confidence": _clean_text(item.get("confidence"), "-"),
                    "evidence_count": _clean_text(item.get("evidence_count"), "0"),
                    "assessment": _clean_text(item.get("assessment"), "需要人工复核。"),
                }
                for index, item in enumerate(steps[:20])
            ],
        },
        "sop_compliance_matrix": extensions["sop_compliance_matrix"],
        "key_action_evidence": extensions["key_action_evidence"],
        "multiview_evidence": extensions["multiview_evidence"],
        "risk_alerts": extensions["risk_alerts"],
        "materials_traceability": extensions["materials_traceability"],
        "overall_assessment": {
            "assessment": "本次分析结果已具备报告化和追溯化基础，但 candidate、inferred、低置信度和告警结果仍需人工复核。",
            "recommendations": [
                {
                    "priority": "high",
                    "recommendation": "优先复核关键动作片段、告警片段和低置信度步骤。",
                    "basis": "这些片段对最终实验合规性和证据完整性影响最大。",
                },
                {
                    "priority": "medium",
                    "recommendation": "保留关键素材索引和原始视频时间点，用于后续查询和审计。",
                    "basis": "素材追溯可以支撑报告结论复核。",
                },
            ],
            "human_review_points": ["candidate/inferred 步骤", "风险告警", "关键动作边界", "对象检测遮挡或缺失片段"],
        },
        "limitations_audit": {
            "limitations": [
                "自动分析结果不能替代最终人工确认。",
                "视频遮挡、低画质、视角缺失或模型漏检会影响证据完整性。",
                reason or "未记录额外回退原因。",
            ],
            "audit_metadata": {
                "report_schema_version": REPORT_SCHEMA_VERSION,
                "generated_at": generated_at,
                "qwen_model": qwen_model,
                "qwen_used": False,
                "analysis_model": _clean_text(metrics.get("model_name"), "-"),
                "result_version": _clean_text(run.get("result_version"), "-"),
                "fallback_reason": reason,
            },
        },
        "signature_page": {
            "system_generated_id": system_generated_id,
            "reviewer": "",
            "approver": "",
            "review_date": "",
            "approval_date": "",
            "statement": "本报告由 RealityLoop 系统自动生成，需经审核人与批准人签署后作为正式归档文件。",
        },
    }
    return base_report


def _extract_dashscope_text(response: Any) -> str:
    output = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
    choices = (output or {}).get("choices") if isinstance(output, dict) else []
    if not choices:
        text = (output or {}).get("text") if isinstance(output, dict) else None
        return str(text or "").strip()
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


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _dashscope_base_url() -> str:
    return (
        os.getenv("DASHSCOPE_BASE_URL")
        or os.getenv("DASHSCOPE_API_BASE_URL")
        or "https://dashscope.aliyuncs.com/api/v1"
    ).replace("/compatible-mode/v1", "/api/v1")


def _call_qwen_report_writer(context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import dashscope  # type: ignore

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY or QWEN_API_KEY is not configured")

    dashscope.base_http_api_url = _dashscope_base_url()
    model = os.getenv("QWEN_REPORT_MODEL", "qwen3.6-max-preview")
    timeout_sec = int(os.getenv("QWEN_REPORT_TIMEOUT_SEC", "120") or "120")
    prompt = f"""你是 RealityLoop 实验室 SOP 态势感知理解平台的专业报告撰写专家。
请基于输入数据撰写一份专业级实验分析 PDF 的结构化内容。

必须遵守：
1. 只输出严格 JSON，不要 Markdown，不要代码块，不要解释。
2. JSON 必须完全遵循给定 schema，字段名不要改。
3. 报告可见标题、正文、结论、建议必须使用正式、克制、可审计的中文。
4. 只有 JSON 字段名、技术 ID 和 key_action_evidence.actions[].objects_en 里的对象名保持英文或原始值；不要翻译对象名。
5. 对 candidate、inferred、低置信度、证据不足的内容必须明确标注需要人工复核。
6. 不夸大结论，不编造输入中不存在的设备、步骤、风险或结果。

标准 JSON Schema：
{json.dumps(STANDARD_REPORT_SCHEMA, ensure_ascii=False, indent=2)}

输入数据：
{json.dumps(context, ensure_ascii=False)[:26000]}
"""
    started = time.perf_counter()
    response = dashscope.Generation.call(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": "你只输出严格 JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        result_format="message",
        timeout=timeout_sec,
    )
    raw = _extract_dashscope_text(response)
    structured = _extract_json_object(raw)
    if not structured:
        raise RuntimeError("Qwen report response did not contain valid JSON")
    return structured, {
        "qwen_used": True,
        "model": model,
        "response_time_ms": int((time.perf_counter() - started) * 1000),
        "raw_response": raw,
    }


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_report(report: Dict[str, Any], context: Dict[str, Any], qwen_meta: Dict[str, Any]) -> Dict[str, Any]:
    context = _enrich_context_from_files(context)
    fallback_reason = _clean_text(qwen_meta.get("error") or qwen_meta.get("fallback_reason"), "")
    normalized = _fallback_report(context, fallback_reason)
    if not isinstance(report, dict):
        return normalized

    for key in [
        "cover",
        "executive_summary",
        "scope",
        "report_reliability",
        "evidence_gallery",
        "procedure_assessment",
        "sop_compliance_matrix",
        "key_action_evidence",
        "multiview_evidence",
        "risk_alerts",
        "materials_traceability",
        "overall_assessment",
        "limitations_audit",
        "signature_page",
    ]:
        if isinstance(report.get(key), dict):
            normalized[key].update(report[key])

    extensions = _report_extensions_from_context(context)
    for key in [
        "report_reliability",
        "evidence_gallery",
        "sop_compliance_matrix",
        "key_action_evidence",
        "multiview_evidence",
        "risk_alerts",
        "materials_traceability",
    ]:
        if isinstance(extensions.get(key), dict):
            normalized[key] = extensions[key]

    for key in ["key_findings"]:
        if isinstance(report.get(key), list):
            normalized[key] = report[key]

    normalized["schema_version"] = REPORT_SCHEMA_VERSION
    cover = _as_dict(normalized.get("cover"))
    cover.setdefault("generated_at", datetime.now().astimezone().isoformat(timespec="seconds"))
    cover["qwen_model"] = _clean_text(qwen_meta.get("model"), cover.get("qwen_model") or os.getenv("QWEN_REPORT_MODEL", "qwen3.6-max-preview"))

    audit = _as_dict(_as_dict(normalized.get("limitations_audit")).get("audit_metadata"))
    audit["report_schema_version"] = REPORT_SCHEMA_VERSION
    audit["qwen_model"] = cover["qwen_model"]
    audit["qwen_used"] = bool(qwen_meta.get("qwen_used"))
    audit["fallback_reason"] = _clean_text(qwen_meta.get("error") or audit.get("fallback_reason"), "")
    normalized["limitations_audit"]["audit_metadata"] = audit

    context_actions = [item for item in _safe_list(context.get("key_actions")) if isinstance(item, dict)]
    context_actions_by_id = {
        _clean_text(item.get("action_id"), ""): item for item in context_actions if _clean_text(item.get("action_id"), "")
    }
    report_actions = _safe_list(_as_dict(normalized.get("key_action_evidence")).get("actions"))
    for index, action in enumerate(report_actions):
        if not isinstance(action, dict):
            continue
        source = context_actions_by_id.get(_clean_text(action.get("action_id"), ""))
        if source is None and index < len(context_actions):
            source = context_actions[index]
        if not source:
            continue
        time_range = _clean_text(action.get("time_range"), "")
        if not time_range or re.fullmatch(r"[-\ss.]+", time_range):
            action["time_range"] = _clean_text(source.get("time_range"), "-")
        if source.get("global_time_range"):
            action.setdefault("global_time_range", source.get("global_time_range"))
        source_objects = _safe_list(source.get("objects_en"))
        objects = [_clean_object_label(item) for item in _safe_list(action.get("objects_en"))]
        objects = [item for item in objects if item]
        if source_objects:
            action["objects_en"] = source_objects[:8]
        elif objects:
            action["objects_en"] = objects[:8]
    return normalized


def _font_path(bold: bool = False) -> Optional[str]:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = _font_path(bold=bold)
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _default_logo_path() -> Optional[Path]:
    candidate = Path(__file__).resolve().parents[2] / "frontend-app" / "public" / "realityloop-logo.png"
    return candidate if candidate.exists() else None


class _PdfPainter:
    def __init__(self, title: str, logo_path: Optional[Path] = None):
        self.width = 1240
        self.height = 1754
        self.margin_x = 86
        self.margin_top = 96
        self.margin_bottom = 86
        self.title = title
        self.logo_path = logo_path or _default_logo_path()
        self.pages: List[Image.Image] = []
        self.page: Image.Image
        self.draw: ImageDraw.ImageDraw
        self.y = 0
        self.page_no = 0
        self.font_title = _font(42, bold=True)
        self.font_h1 = _font(30, bold=True)
        self.font_h2 = _font(24, bold=True)
        self.font_body = _font(20)
        self.font_small = _font(16)
        self.font_tiny = _font(14)
        self.new_page()

    def new_page(self) -> None:
        self.page_no += 1
        self.page = Image.new("RGB", (self.width, self.height), "#ffffff")
        self.draw = ImageDraw.Draw(self.page)
        self.pages.append(self.page)
        self.y = self.margin_top
        self._draw_header_footer()

    def _draw_header_footer(self) -> None:
        self.draw.rectangle((0, 0, self.width, 68), fill="#f8fafc")
        x = self.margin_x
        if self.logo_path and self.logo_path.exists():
            try:
                logo = Image.open(self.logo_path).convert("RGBA")
                logo.thumbnail((42, 42))
                self.page.paste(logo, (self.margin_x, 14), logo)
                x = self.margin_x + 54
            except Exception:
                x = self.margin_x
        self.draw.text((x, 20), "RealityLoop 实验室SOP态势感知理解平台", font=self.font_small, fill="#0f172a")
        self.draw.text((self.width - self.margin_x - 120, self.height - 50), f"Page {self.page_no}", font=self.font_tiny, fill="#64748b")
        self.draw.line((self.margin_x, self.height - 68, self.width - self.margin_x, self.height - 68), fill="#e2e8f0", width=1)

    def ensure(self, needed: int) -> None:
        if self.y + needed > self.height - self.margin_bottom:
            self.new_page()

    def _font_size(self, font: ImageFont.ImageFont) -> int:
        return int(getattr(font, "size", 20))

    def text_width(self, text: str, font: ImageFont.ImageFont) -> float:
        return float(self.draw.textlength(text, font=font))

    def wrap(self, text: str, font: ImageFont.ImageFont, width: int) -> List[str]:
        lines: List[str] = []
        for paragraph in str(text or "").splitlines() or [""]:
            paragraph = paragraph.strip()
            if not paragraph:
                lines.append("")
                continue
            current = ""
            for char in paragraph:
                candidate = current + char
                if self.text_width(candidate, font) <= width or not current:
                    current = candidate
                else:
                    lines.append(current)
                    current = char
            if current:
                lines.append(current)
        return lines

    def paragraph(self, text: str, *, font: Optional[ImageFont.ImageFont] = None, fill: str = "#334155", spacing: int = 10) -> None:
        font = font or self.font_body
        lines = self.wrap(text, font, self.width - self.margin_x * 2)
        line_h = self._font_size(font) + 10
        self.ensure(max(42, len(lines) * line_h + spacing))
        for line in lines:
            self.draw.text((self.margin_x, self.y), line, font=font, fill=fill)
            self.y += line_h
        self.y += spacing

    def heading(self, text: str, level: int = 1) -> None:
        font = self.font_h1 if level == 1 else self.font_h2
        self.ensure(72)
        if level == 1:
            self.draw.rectangle((self.margin_x, self.y - 4, self.margin_x + 8, self.y + 38), fill="#2563eb")
            self.draw.text((self.margin_x + 20, self.y), text, font=font, fill="#0f172a")
        else:
            self.draw.text((self.margin_x, self.y), text, font=font, fill="#0f172a")
            self.draw.line((self.margin_x, self.y + 40, self.width - self.margin_x, self.y + 40), fill="#e2e8f0", width=2)
        self.y += 58

    def metric_cards(self, items: Sequence[Tuple[str, str]]) -> None:
        cols = 3
        gap = 18
        card_w = (self.width - self.margin_x * 2 - gap * (cols - 1)) // cols
        card_h = 104
        rows = (len(items) + cols - 1) // cols
        self.ensure(rows * (card_h + gap) + 12)
        for index, (label, value) in enumerate(items):
            col = index % cols
            row = index // cols
            x = self.margin_x + col * (card_w + gap)
            y = self.y + row * (card_h + gap)
            self.draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=10, fill="#f8fafc", outline="#dbeafe", width=2)
            self.draw.text((x + 22, y + 18), label, font=self.font_small, fill="#475569")
            value_text = _clean_text(value, "-")
            value_font = self.font_h2
            if self.text_width(value_text, value_font) > card_w - 44:
                value_font = self.font_small
            value_lines = self.wrap(value_text, value_font, card_w - 44)[:2]
            line_h = self._font_size(value_font) + 6
            for line_index, line in enumerate(value_lines):
                self.draw.text((x + 22, y + 50 + line_index * line_h), line, font=value_font, fill="#0f172a")
        self.y += rows * (card_h + gap) + 12

    def bullet_list(self, items: Sequence[Any]) -> None:
        for item in items:
            text = _clean_text(item, "")
            if not text:
                continue
            lines = self.wrap(text, self.font_body, self.width - self.margin_x * 2 - 30)
            line_h = self._font_size(self.font_body) + 10
            self.ensure(max(40, len(lines) * line_h + 6))
            self.draw.ellipse((self.margin_x + 2, self.y + 10, self.margin_x + 12, self.y + 20), fill="#2563eb")
            for line in lines:
                self.draw.text((self.margin_x + 30, self.y), line, font=self.font_body, fill="#334155")
                self.y += line_h
            self.y += 6

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]], widths: Sequence[int]) -> None:
        x0 = self.margin_x
        line_h = self._font_size(self.font_small) + 8
        header_h = 44
        self.ensure(header_h + 16)
        self.draw.rounded_rectangle((x0, self.y, x0 + sum(widths), self.y + header_h), radius=8, fill="#eff6ff", outline="#bfdbfe")
        x = x0
        for header, width in zip(headers, widths):
            self.draw.text((x + 10, self.y + 12), header, font=self.font_small, fill="#1e40af")
            x += width
        self.y += header_h
        for row in rows:
            wrapped_cells = [
                self.wrap(_clean_text(cell, "-"), self.font_small, max(40, width - 18))
                for cell, width in zip(row, widths)
            ]
            row_h = max(line_h * max(1, len(lines)) for lines in wrapped_cells) + 22
            self.ensure(row_h + 4)
            self.draw.rectangle((x0, self.y, x0 + sum(widths), self.y + row_h), fill="#ffffff", outline="#e2e8f0")
            x = x0
            for lines, width in zip(wrapped_cells, widths):
                cy = self.y + 10
                for line in lines:
                    self.draw.text((x + 10, cy), line, font=self.font_small, fill="#334155")
                    cy += line_h
                x += width
            self.y += row_h
        self.y += 18

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.pages:
            self.new_page()
        first, rest = self.pages[0], self.pages[1:]
        first.save(output_path, "PDF", save_all=True, append_images=rest, resolution=150.0)


def _metric_pairs(report: Dict[str, Any]) -> List[Tuple[str, str]]:
    metrics = _safe_list(_as_dict(report.get("executive_summary")).get("key_metrics"))
    pairs: List[Tuple[str, str]] = []
    for item in metrics:
        if isinstance(item, dict):
            pairs.append((_clean_text(item.get("label"), "-"), _clean_text(item.get("value"), "-")))
    return pairs[:9]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _report_template_dir() -> Path:
    return _project_root() / "templates" / "reports"


def _image_data_uri(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "image/svg+xml" if suffix == "svg" else f"image/{suffix}"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_preview_data_uri(value: Any, max_size: Tuple[int, int] = (1100, 760)) -> str:
    path_text = _path_if_exists(value)
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        image = Image.open(path)
        image.thumbnail(max_size)
        if image.mode in {"RGBA", "LA"}:
            canvas = Image.new("RGB", image.size, "#ffffff")
            canvas.paste(image, mask=image.getchannel("A"))
            image = canvas
        else:
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return _image_data_uri(path)


def _render_report_html(
    *,
    report: Dict[str, Any],
    context: Dict[str, Any],
    logo_path: Optional[Path],
    qwen_meta: Dict[str, Any],
) -> str:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(_report_template_dir())),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("professional_report.html.j2")
    return template.render(
        report=report,
        context=context,
        qwen_meta=qwen_meta,
        schema_version=REPORT_SCHEMA_VERSION,
        sections=STANDARD_REPORT_SECTIONS,
        logo_data_uri=_image_data_uri(logo_path or _default_logo_path()),
        image_data_uri=_image_preview_data_uri,
    )


def _normalize_windows_path(value: str) -> str:
    return ntpath.normcase(ntpath.normpath(value.strip().strip('"')))


def _ensure_weasyprint_runtime_path() -> None:
    if os.name != "nt":
        return
    preferred_candidates = [
        Path(r"C:\Program Files\GTK3-Runtime Win64\bin"),
        Path(r"C:\Program Files\GTK3-Runtime\bin"),
        Path(r"C:\msys64\mingw64\bin"),
    ]
    blocked_candidates = [
        Path(r"C:\Program Files\Gtk-Runtime\bin"),
    ]
    preferred = next((path for path in preferred_candidates if path.exists()), None)
    preferred_str = str(preferred) if preferred is not None else ""
    preferred_norm = _normalize_windows_path(preferred_str) if preferred_str else ""
    known_norms = {
        _normalize_windows_path(str(path))
        for path in [*preferred_candidates, *blocked_candidates]
    }
    blocked_norms = {_normalize_windows_path(str(path)) for path in blocked_candidates}
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep)
    remaining: List[str] = []
    seen_norms = set()
    for part in parts:
        cleaned = part.strip().strip('"')
        if not cleaned:
            continue
        norm = _normalize_windows_path(cleaned)
        if norm in blocked_norms or norm in known_norms or norm in seen_norms:
            continue
        seen_norms.add(norm)
        remaining.append(cleaned)
    if preferred is None:
        os.environ["PATH"] = os.pathsep.join(remaining)
        return
    os.environ["PATH"] = os.pathsep.join([preferred_str, *remaining])
    if hasattr(os, "add_dll_directory") and preferred_norm not in _WEASYPRINT_DLL_DIRECTORY_PATHS:
        try:
            handle = os.add_dll_directory(preferred_str)
        except OSError:
            return
        _WEASYPRINT_DLL_DIRECTORY_HANDLES.append(handle)
        _WEASYPRINT_DLL_DIRECTORY_PATHS.append(preferred_norm)


def _render_pdf_report_with_weasyprint(
    *,
    report: Dict[str, Any],
    context: Dict[str, Any],
    output_path: Path,
    logo_path: Optional[Path],
    qwen_meta: Dict[str, Any],
) -> None:
    _ensure_weasyprint_runtime_path()
    from weasyprint import HTML

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _render_report_html(report=report, context=context, logo_path=logo_path, qwen_meta=qwen_meta)
    output_path.with_suffix(".html").write_text(html, encoding="utf-8")
    HTML(string=html, base_url=str(_project_root())).write_pdf(str(output_path))


def _draw_cover_brand(painter: _PdfPainter, label: str) -> None:
    x = painter.margin_x
    y = painter.y
    text_x = x
    if painter.logo_path and painter.logo_path.exists():
        try:
            logo = Image.open(painter.logo_path).convert("RGBA")
            logo.thumbnail((54, 54))
            painter.page.paste(logo, (x, y), logo)
            text_x = x + 68
        except Exception:
            text_x = x
    painter.draw.text((text_x, y + 12), label, font=painter.font_body, fill="#2563eb")
    painter.y += 72


def _signature_box(painter: _PdfPainter, title: str, value: str, date_label: str) -> None:
    x = painter.margin_x
    y = painter.y
    width = painter.width - painter.margin_x * 2
    height = 150
    painter.ensure(height + 26)
    painter.draw.rounded_rectangle((x, y, x + width, y + height), radius=10, fill="#ffffff", outline="#cbd5e1", width=2)
    painter.draw.text((x + 24, y + 22), title, font=painter.font_h2, fill="#0f172a")
    painter.draw.text((x + 220, y + 28), _clean_text(value, "____________________________"), font=painter.font_body, fill="#334155")
    line_y = y + 82
    painter.draw.line((x + 220, line_y, x + width - 36, line_y), fill="#94a3b8", width=2)
    painter.draw.text((x + 24, y + 102), date_label, font=painter.font_body, fill="#475569")
    painter.draw.line((x + 220, y + 124, x + width - 36, y + 124), fill="#94a3b8", width=2)
    painter.y += height + 26


def _render_signature_page(painter: _PdfPainter, report: Dict[str, Any]) -> None:
    signature = _as_dict(report.get("signature_page"))
    painter.new_page()
    painter.heading("十、签字页")
    painter.paragraph(_clean_text(signature.get("statement"), "本报告由 RealityLoop 系统自动生成，需经审核人与批准人签署后作为正式归档文件。"))
    painter.metric_cards(
        [
            ("系统生成标识", _clean_text(signature.get("system_generated_id"), "-")),
            ("审核人", _clean_text(signature.get("reviewer"), "待签署")),
            ("批准人", _clean_text(signature.get("approver"), "待签署")),
        ]
    )
    _signature_box(painter, "审核人", _clean_text(signature.get("reviewer"), ""), "审核日期")
    _signature_box(painter, "批准人", _clean_text(signature.get("approver"), ""), "批准日期")
    painter.paragraph("系统生成标识用于追溯报告生成批次、实验运行和审计记录。签署前请完成关键动作、风险告警和低置信度结果复核。", fill="#64748b")


def _dict_rows(items: Sequence[Dict[str, Any]], fields: Sequence[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for item in items:
        row = []
        for field in fields:
            value = item.get(field)
            if isinstance(value, list):
                value = ", ".join(_clean_text(part, "") for part in value if _clean_text(part, ""))
            row.append(_clean_text(value, "-"))
        rows.append(row)
    return rows


def render_pdf_report(
    *,
    report: Dict[str, Any],
    context: Dict[str, Any],
    output_path: Path,
    logo_path: Optional[Path] = None,
    qwen_meta: Optional[Dict[str, Any]] = None,
) -> None:
    qwen_meta = qwen_meta or {}
    context = _enrich_context_from_files(context)
    report = _normalize_report(report, context, qwen_meta)
    try:
        _render_pdf_report_with_weasyprint(
            report=report,
            context=context,
            output_path=output_path,
            logo_path=logo_path,
            qwen_meta=qwen_meta,
        )
        return
    except Exception as exc:
        qwen_meta["pdf_renderer"] = "pillow_fallback"
        qwen_meta["weasyprint_error"] = str(exc)
        try:
            html = _render_report_html(report=report, context=context, logo_path=logo_path, qwen_meta=qwen_meta)
            output_path.with_suffix(".html").write_text(html, encoding="utf-8")
        except Exception:
            pass

    cover = _as_dict(report.get("cover"))
    title = _clean_text(cover.get("report_title"), "实验分析专业报告")
    painter = _PdfPainter(title=title, logo_path=logo_path)

    painter.ensure(360)
    painter.y += 90
    _draw_cover_brand(painter, "RealityLoop 专业实验分析报告")
    painter.paragraph(title, font=painter.font_title, fill="#0f172a", spacing=30)
    painter.metric_cards(
        [
            ("实验名称", _clean_text(cover.get("experiment_name"), "-")),
            ("实验ID", _clean_text(cover.get("experiment_id"), "-")),
            ("运行ID", _clean_text(cover.get("run_id"), "-")),
            ("结果版本", _clean_text(cover.get("result_version"), "-")),
            ("生成时间", _clean_text(cover.get("generated_at"), "-")),
            ("报告结构版本", REPORT_SCHEMA_VERSION),
        ]
    )
    painter.paragraph("本报告由 RealityLoop 自动分析结果与标准化报告结构生成。候选、推断和低置信度内容均需结合原始视频进行人工复核。")

    executive = _as_dict(report.get("executive_summary"))
    painter.heading("一、执行摘要")
    painter.paragraph(_clean_text(executive.get("overall_conclusion"), "-"), font=painter.font_h2, fill="#0f172a")
    painter.paragraph(_clean_text(executive.get("summary"), "-"))
    painter.paragraph(f"证据充分性：{_clean_text(executive.get('evidence_sufficiency'), '-')}")
    metrics = _metric_pairs(report)
    if metrics:
        painter.metric_cards(metrics)

    scope = _as_dict(report.get("scope"))
    painter.heading("二、范围与数据来源")
    painter.paragraph(_clean_text(scope.get("description"), "-"))
    painter.paragraph("分析模块：", font=painter.font_body, fill="#0f172a", spacing=4)
    painter.bullet_list(_safe_list(scope.get("analysis_modules")))
    painter.paragraph("数据来源：", font=painter.font_body, fill="#0f172a", spacing=4)
    painter.bullet_list(_safe_list(scope.get("data_sources")))

    findings = [item for item in _safe_list(report.get("key_findings")) if isinstance(item, dict)]
    painter.heading("三、关键结论")
    if findings:
        rows = _dict_rows(findings, ["finding", "evidence", "impact", "confidence"])
        painter.table(["发现", "证据", "影响", "置信/复核"], rows, [270, 330, 270, 210])
    else:
        painter.paragraph("未形成明确关键结论。")

    procedure = _as_dict(report.get("procedure_assessment"))
    painter.heading("四、步骤执行评估")
    painter.paragraph(_clean_text(procedure.get("summary"), "-"))
    step_items = [item for item in _safe_list(procedure.get("steps")) if isinstance(item, dict)]
    if step_items:
        painter.table(
            ["序号", "步骤", "状态", "时间", "置信度", "评估"],
            _dict_rows(step_items, ["index", "step_name", "status", "time_range", "confidence", "assessment"]),
            [70, 250, 120, 160, 100, 380],
        )
    else:
        painter.paragraph("未形成结构化步骤。")

    key_action_evidence = _as_dict(report.get("key_action_evidence"))
    painter.heading("五、关键动作证据")
    painter.paragraph(_clean_text(key_action_evidence.get("summary"), "-"))
    action_items = [item for item in _safe_list(key_action_evidence.get("actions")) if isinstance(item, dict)]
    if action_items:
        painter.table(
            ["ID", "动作类型", "时间", "对象(英文)", "证据摘要", "复核状态"],
            _dict_rows(action_items, ["action_id", "action_type", "time_range", "objects_en", "evidence_summary", "review_status"]),
            [110, 150, 160, 220, 270, 170],
        )
    else:
        painter.paragraph("未形成关键动作片段。")

    risks = _as_dict(report.get("risk_alerts"))
    painter.heading("六、风险与异常")
    painter.paragraph(_clean_text(risks.get("summary"), "-"))
    risk_items = [item for item in _safe_list(risks.get("alerts")) if isinstance(item, dict)]
    if risk_items:
        painter.table(
            ["等级", "规则/风险", "时间/帧", "证据", "建议"],
            _dict_rows(risk_items, ["severity", "rule", "time_or_frame", "evidence", "recommendation"]),
            [90, 230, 130, 320, 310],
        )
    else:
        painter.paragraph("未发现明确风险告警。")

    materials_traceability = _as_dict(report.get("materials_traceability"))
    painter.heading("七、关键素材与追溯")
    painter.paragraph(_clean_text(materials_traceability.get("summary"), "-"))
    material_items = [item for item in _safe_list(materials_traceability.get("materials")) if isinstance(item, dict)]
    if material_items:
        painter.table(
            ["素材", "事件类型", "时间", "证据等级", "相关对象(英文)"],
            _dict_rows(material_items, ["material_name", "event_type", "time_range", "evidence_grade", "related_objects_en"]),
            [280, 180, 160, 140, 320],
        )
    else:
        painter.paragraph("未发布关键素材索引。")

    assessment = _as_dict(report.get("overall_assessment"))
    painter.heading("八、综合评估与建议")
    painter.paragraph(_clean_text(assessment.get("assessment"), "-"))
    recommendations = [item for item in _safe_list(assessment.get("recommendations")) if isinstance(item, dict)]
    if recommendations:
        painter.table(
            ["优先级", "建议", "依据"],
            _dict_rows(recommendations, ["priority", "recommendation", "basis"]),
            [120, 520, 440],
        )
    painter.paragraph("人工复核重点：", font=painter.font_body, fill="#0f172a", spacing=4)
    painter.bullet_list(_safe_list(assessment.get("human_review_points")))

    audit_section = _as_dict(report.get("limitations_audit"))
    painter.heading("九、局限性与审计信息")
    painter.bullet_list(_safe_list(audit_section.get("limitations")))
    audit = _as_dict(audit_section.get("audit_metadata"))
    audit_rows = [[key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, bool)) else _clean_text(value, "-")] for key, value in audit.items()]
    if audit_rows:
        painter.table(["字段", "值"], audit_rows, [320, 760])

    _render_signature_page(painter, report)
    painter.save(output_path)


def generate_professional_report_pdf(
    *,
    overview: Dict[str, Any],
    key_actions: Optional[Dict[str, Any]],
    materials: Dict[str, Any],
    output_pdf_path: Path,
    logo_path: Optional[Path] = None,
) -> Dict[str, Any]:
    context = _enrich_context_from_files(build_report_context(overview=overview, key_actions=key_actions, materials=materials))
    try:
        report, qwen_meta = _call_qwen_report_writer(context)
    except Exception as exc:
        qwen_meta = {
            "qwen_used": False,
            "model": os.getenv("QWEN_REPORT_MODEL", "qwen3.6-max-preview"),
            "error": str(exc),
        }
        report = _fallback_report(context, str(exc))

    report = _normalize_report(report, context, qwen_meta)
    render_pdf_report(
        report=report,
        context=context,
        output_path=output_pdf_path,
        logo_path=logo_path,
        qwen_meta=qwen_meta,
    )
    sidecar = output_pdf_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "pdf_path": str(output_pdf_path),
                "context": context,
                "report": report,
                "qwen": qwen_meta,
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "pdf_path": str(output_pdf_path),
        "sidecar_path": str(sidecar),
        "qwen": qwen_meta,
    }


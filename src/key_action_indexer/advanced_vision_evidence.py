from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from .model_observations import load_or_build_model_observation_events
from .model_inventory import discover_lab_assets, load_model_inventory
from .schemas import read_jsonl, write_jsonl


EVIDENCE_FIELDS = (
    "evidence_id",
    "session_id",
    "evidence_type",
    "segment_id",
    "micro_segment_id",
    "global_start_time",
    "global_end_time",
    "object_label",
    "action_type",
    "visual_confirmation_level",
    "confirmation_level",
    "confidence",
    "confidence_reasons",
    "evidence_reasons",
    "limitations",
    "multiview_consistency",
    "metrics",
    "asset_refs",
    "evidence_refs",
    "requires_human_confirmation",
    "payload",
)

CONTAINER_TOKENS = ("bottle", "tube", "vial", "flask", "beaker", "container", "瓶", "试管", "容器")
PANEL_TOKENS = ("balance", "panel", "display", "button", "knob", "scale", "天平", "面板", "读数")
LIQUID_TOKENS = ("pipette", "tube", "liquid", "transfer", "pipetting", "sample_adding", "移液", "加样", "微升")
CAP_LID_TOKENS = ("cap", "lid", "tube_cap", "tube-cap", "bottle_cap", "盖", "瓶盖")
CAP_LID_TOKENS = ("cap", "lid", "tube_cap", "tube-cap", "bottle_cap", "盖", "瓶盖")


def build_advanced_vision_evidence(session_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    target = Path(output_path) if output_path is not None else metadata / "advanced_vision_evidence.jsonl"
    summary_path = target.with_name("advanced_vision_evidence_summary.json")
    micro_rows = _read_jsonl_if_exists(metadata / "micro_segments.jsonl")
    asset_rows = _read_jsonl_if_exists(metadata / "material_asset_catalog.jsonl")
    yolo_rows = _load_yolo_rows(session)
    assets_by_micro = _assets_by_micro(asset_rows)
    model_observation_rows, model_observation_summary = load_or_build_model_observation_events(session)
    inventory = _load_or_discover_inventory(session)

    evidence: list[dict[str, Any]] = []
    for index, micro in enumerate(micro_rows, start=1):
        if not isinstance(micro, Mapping):
            continue
        micro_id = str(micro.get("micro_segment_id") or f"micro_{index:06d}")
        scoped_rows = _rows_for_micro(yolo_rows, micro)
        evidence.extend(_hand_object_contact_evidence(micro, scoped_rows, assets_by_micro.get(micro_id, [])))
        evidence.extend(_trajectory_evidence(micro, scoped_rows, assets_by_micro.get(micro_id, [])))
        evidence.extend(_image_pair_evidence(session, micro, assets_by_micro.get(micro_id, []), inventory))
        evidence.extend(_panel_evidence(session, micro, assets_by_micro.get(micro_id, [])))
    evidence.extend(_model_observation_evidence(model_observation_rows))
    evidence = _prefer_model_observation_upgrades(evidence)

    evidence.sort(key=lambda row: (str(row.get("global_start_time") or ""), str(row.get("evidence_id") or "")))
    write_jsonl(target, [_stable(row) for row in evidence])
    type_counts = Counter(str(row.get("evidence_type") or "unknown") for row in evidence)
    level_counts = Counter(str(row.get("visual_confirmation_level") or "unknown") for row in evidence)
    summary = {
        "session_id": _first_text(
            [
                *(row.get("session_id") for row in evidence if isinstance(row, Mapping)),
                *(row.get("session_id") for row in model_observation_rows if isinstance(row, Mapping)),
                *(row.get("session_id") for row in micro_rows if isinstance(row, Mapping)),
                _manifest_session_id(session),
            ]
        ),
        "evidence_count": len(evidence),
        "evidence_type_counts": dict(sorted(type_counts.items())),
        "visual_confirmation_level_counts": dict(sorted(level_counts.items())),
        "input_counts": {
            "micro_segments": len(micro_rows),
            "yolo_frame_rows": len(yolo_rows),
            "material_assets": len(asset_rows),
            "model_observation_events": len(model_observation_rows),
            "model_observation_inputs": model_observation_summary.get("input_counts", {}),
        },
        "real_model_inventory": {
            "primary_model": inventory.get("primary_model", {}),
            "model_count": inventory.get("model_count", 0),
            "dataset_count": inventory.get("dataset_count", 0),
            "capabilities": inventory.get("capabilities", {}),
        },
        "advanced_vision_evidence": str(target),
        "summary_path": str(summary_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_advanced_vision_evidence(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    return read_jsonl(source) if source.exists() else []


def _hand_object_contact_evidence(
    micro: Mapping[str, Any],
    yolo_rows: list[Mapping[str, Any]],
    assets: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    interaction = _as_dict(micro.get("interaction"))
    primary = str(interaction.get("primary_object") or "")
    interaction_type = str(interaction.get("interaction_type") or (f"hand_{primary}_contact" if primary else "hand_object_contact"))
    if not primary and not interaction_type:
        return []
    row_matches: list[Mapping[str, Any]] = []
    scores: list[float] = []
    overlaps: list[float] = []
    distances: list[float] = []
    for row in yolo_rows:
        for candidate in row.get("hand_object_interactions") or []:
            if not isinstance(candidate, Mapping):
                continue
            object_label = str(candidate.get("object_label") or "")
            if primary and object_label and object_label != primary:
                continue
            row_matches.append(row)
            scores.append(_float(candidate.get("score", candidate.get("confidence")), 0.0))
            overlaps.append(_float(candidate.get("iou", candidate.get("bbox_overlap")), 0.0))
            distance = _float(candidate.get("distance_px"), None)
            if distance is not None:
                distances.append(distance)
    if not scores and interaction.get("max_interaction_score") is not None:
        scores.append(_float(interaction.get("max_interaction_score"), 0.0))
    if not scores and not primary:
        return []
    max_score = max(scores or [0.0])
    avg_score = sum(scores) / len(scores) if scores else 0.0
    frame_count = len(scores)
    multiview = _multiview_consistency(row_matches, object_label=primary)
    confirmed = max_score >= 0.65 and (frame_count >= 2 or str(interaction.get("primary_object_arbitration") or "").startswith("tracklet"))
    visual_level = "hand_object_contact_confirmed" if confirmed else "hand_object_contact_candidate"
    limitations = []
    if frame_count < 2:
        limitations.append("single-frame or metadata-only hand-object contact evidence")
    if multiview.get("status") != "consistent_multiview":
        limitations.append("not confirmed by both views")
    confidence = min(0.93, 0.35 + max_score * 0.45 + min(frame_count, 6) * 0.03)
    return [
        _event(
            micro,
            evidence_type="hand_object_contact",
            object_label=primary,
            visual_confirmation_level=visual_level,
            confidence=confidence,
            evidence_reasons=[
                f"interaction_type={interaction_type}",
                f"max_interaction_score={max_score:.3f}",
                f"supporting_frame_count={frame_count}",
            ],
            limitations=limitations,
            metrics={
                "interaction_type": interaction_type,
                "avg_interaction_score": round(avg_score, 4),
                "max_interaction_score": round(max_score, 4),
                "supporting_frame_count": frame_count,
                "max_bbox_overlap": round(max(overlaps or [0.0]), 4),
                "avg_distance_px": round(sum(distances) / len(distances), 4) if distances else None,
                "primary_object_arbitration": interaction.get("primary_object_arbitration"),
                "primary_object_vote_score": interaction.get("primary_object_vote_score"),
                "primary_object_vote_margin": interaction.get("primary_object_vote_margin"),
            },
            asset_refs=_asset_refs(assets),
            payload={"interaction": dict(interaction)},
            multiview_consistency=multiview,
            evidence_refs=[*_asset_refs(assets), *_row_evidence_refs(row_matches[:8])],
        )
    ]


def _trajectory_evidence(micro: Mapping[str, Any], yolo_rows: list[Mapping[str, Any]], assets: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for row_index, row in enumerate(yolo_rows):
        frame_key = _frame_key(row, row_index)
        for det in _detections(row):
            label = str(det.get("label") or det.get("object_label") or "").strip()
            bbox = _bbox(det.get("bbox"))
            if not label or label in {"hand", "gloved_hand"} or bbox is None:
                continue
            detections.append(
                {
                    "label": label,
                    "time_sec": _row_time(row),
                    "global_time": row.get("global_time"),
                    "bbox": bbox,
                    "confidence": _float(det.get("confidence"), 0.0),
                    "track_id": _first_non_empty(det.get("track_id"), det.get("object_track_id"), det.get("tracklet_id")),
                    "track_source": _first_non_empty(det.get("track_source"), det.get("tracker"), row.get("track_source")),
                    "frame_key": frame_key,
                    "view": row.get("view") or row.get("source_view") or row.get("camera"),
                }
            )
    same_class_per_frame = _same_class_per_frame(detections)
    tracks_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in _tracks_from_detections(detections):
        tracks_by_label[str(track.get("label") or "")].append(track)
    events = []
    primary = str(_as_dict(micro.get("interaction")).get("primary_object") or "")
    candidate_labels = [primary] if primary in tracks_by_label else list(tracks_by_label)
    for label in candidate_labels:
        label_tracks = sorted(tracks_by_label.get(label, []), key=lambda item: len(item.get("points") or []), reverse=True)
        for item in label_tracks:
            track = sorted(item.get("points") or [], key=lambda row: row["time_sec"])
            if len(track) < 3:
                continue
            label_context = {
                "same_class_track_count": len(label_tracks),
                "max_same_class_per_frame": same_class_per_frame.get(label, 0),
                "explicit_track_count": sum(1 for candidate in label_tracks if _explicit_track_identity(candidate)),
                "inferred_track_count": sum(1 for candidate in label_tracks if not _explicit_track_identity(candidate)),
            }
            event = _trajectory_event(micro, label, item, track, assets, label_context)
            if event:
                events.append(event)
                break
    return events


def _trajectory_event(
    micro: Mapping[str, Any],
    label: str,
    track_item: Mapping[str, Any],
    track: list[dict[str, Any]],
    assets: list[Mapping[str, Any]],
    label_context: Mapping[str, Any],
) -> dict[str, Any] | None:
    first = track[0]
    last = track[-1]
    first_center = _center(first["bbox"])
    last_center = _center(last["bbox"])
    displacement = math.dist(first_center, last_center)
    widths = [abs(row["bbox"][2] - row["bbox"][0]) for row in track]
    heights = [abs(row["bbox"][3] - row["bbox"][1]) for row in track]
    scale = max(median(widths + heights), 1.0)
    normalized = displacement / scale
    path_length = sum(math.dist(_center(a["bbox"]), _center(b["bbox"])) for a, b in zip(track, track[1:]))
    if normalized < 0.25 and path_length / scale < 0.5:
        return None
    confidence = min(0.95, 0.45 + min(normalized, 2.0) * 0.18 + min(len(track), 20) * 0.01)
    source = str(track_item.get("track_source") or "bbox_temporal_association")
    identity_confidence = _track_identity_confidence(track_item, track, label_context)
    identity_risk_reasons = _track_identity_risk_reasons(track_item, label_context, identity_confidence)
    visual_level = "trajectory_confirmed" if identity_confidence >= 0.65 else "trajectory_candidate_identity_risk"
    if identity_risk_reasons:
        confidence = min(confidence, max(0.45, 0.52 + identity_confidence * 0.18))
    limitations = []
    if not _explicit_track_identity(track_item):
        limitations.append("track identity inferred from bbox continuity; use model track_id/ByteTrack for crowded same-class scenes")
    if identity_risk_reasons:
        limitations.append("same-class multi-target identity risk: " + "; ".join(identity_risk_reasons))
    return _event(
        micro,
        evidence_type="object_trajectory_movement",
        object_label=label,
        visual_confirmation_level=visual_level,
        confidence=confidence,
        evidence_reasons=[
            f"tracked {label} across {len(track)} detections using {source}",
            f"normalized_displacement={normalized:.3f}",
            f"path_length_px={path_length:.1f}",
            f"identity_confidence={identity_confidence:.3f}",
        ],
        limitations=limitations,
        metrics={
            "track_id": track_item.get("track_id"),
            "track_source": source,
            "identity_confidence": identity_confidence,
            "identity_source": track_item.get("identity_source"),
            "identity_status": "stable" if identity_confidence >= 0.65 else "same_class_risk",
            "same_class_track_count": int(label_context.get("same_class_track_count") or 0),
            "max_same_class_per_frame": int(label_context.get("max_same_class_per_frame") or 0),
            "explicit_track_count": int(label_context.get("explicit_track_count") or 0),
            "inferred_track_count": int(label_context.get("inferred_track_count") or 0),
            "identity_risk_reasons": identity_risk_reasons,
            "avg_detection_confidence": _average_detection_confidence(track),
            "detection_count": len(track),
            "displacement_px": round(displacement, 3),
            "normalized_displacement": round(normalized, 4),
            "path_length_px": round(path_length, 3),
            "first_center": [round(first_center[0], 3), round(first_center[1], 3)],
            "last_center": [round(last_center[0], 3), round(last_center[1], 3)],
        },
        asset_refs=_asset_refs(assets),
        payload={"track": track[:20], "identity_risk_reasons": identity_risk_reasons},
    )


def _tracks_from_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    explicit: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    implicit_points: list[dict[str, Any]] = []
    for item in sorted(detections, key=lambda row: float(row.get("time_sec") or 0.0)):
        track_id = item.get("track_id")
        if track_id is not None and str(track_id) != "":
            explicit[(str(item.get("label") or ""), str(track_id))].append(item)
        else:
            implicit_points.append(item)
    tracks = [
        {
            "label": label,
            "track_id": track_id,
            "track_source": _explicit_track_source(points),
            "identity_source": "explicit_track_id",
            "points": points,
        }
        for (label, track_id), points in explicit.items()
    ]
    active: list[dict[str, Any]] = []
    next_id = 1
    for point in implicit_points:
        label = str(point.get("label") or "")
        best: dict[str, Any] | None = None
        best_score = 999.0
        for track in active:
            if str(track.get("label") or "") != label:
                continue
            points = track.get("points") or []
            if not points:
                continue
            last = points[-1]
            if float(point["time_sec"]) <= float(last["time_sec"]):
                continue
            distance = math.dist(_center(point["bbox"]), _center(last["bbox"]))
            scale = max(_bbox_scale(point["bbox"]), _bbox_scale(last["bbox"]), 1.0)
            score = distance / scale
            if score < best_score:
                best_score = score
                best = track
        if best is not None and best_score <= 2.0:
            best["points"].append(point)
            continue
        track = {
            "label": label,
            "track_id": f"inferred_{next_id:04d}",
            "track_source": "bbox_temporal_association",
            "identity_source": "inferred_bbox_continuity",
            "points": [point],
        }
        next_id += 1
        active.append(track)
    tracks.extend(active)
    return tracks


def _frame_key(row: Mapping[str, Any], row_index: int) -> str:
    for key in ("frame_id", "source_image_path", "image_path", "path"):
        value = row.get(key)
        if value:
            return str(value)
    view = str(row.get("view") or row.get("source_view") or row.get("camera") or "view")
    return f"{view}:{_row_time(row):.3f}:{row_index}"


def _same_class_per_frame(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[tuple[str, str]] = Counter()
    for item in detections:
        label = str(item.get("label") or "")
        frame_key = str(item.get("frame_key") or "")
        if label and frame_key:
            counts[(label, frame_key)] += 1
    max_by_label: dict[str, int] = defaultdict(int)
    for (label, _frame_key_value), count in counts.items():
        max_by_label[label] = max(max_by_label[label], count)
    return dict(max_by_label)


def _explicit_track_source(points: list[dict[str, Any]]) -> str:
    for point in points:
        source = _first_non_empty(point.get("track_source"))
        if source:
            return str(source)
    return "model_track_id"


def _explicit_track_identity(track_item: Mapping[str, Any]) -> bool:
    return str(track_item.get("identity_source") or "") == "explicit_track_id"


def _track_identity_confidence(
    track_item: Mapping[str, Any],
    track: list[dict[str, Any]],
    label_context: Mapping[str, Any],
) -> float:
    explicit_identity = _explicit_track_identity(track_item)
    score = 0.9 if explicit_identity else 0.72
    avg_detection_confidence = _average_detection_confidence(track)
    if avg_detection_confidence is not None:
        score += (avg_detection_confidence - 0.5) * 0.12
    if len(track) >= 6:
        score += 0.03
    same_class_track_count = max(0, int(label_context.get("same_class_track_count") or 0) - 1)
    max_same_class_per_frame = max(0, int(label_context.get("max_same_class_per_frame") or 0) - 1)
    score -= min(0.25, same_class_track_count * 0.08)
    score -= min(0.25, max_same_class_per_frame * 0.12)
    if not explicit_identity and same_class_track_count:
        score -= 0.08
    return round(max(0.05, min(0.98, score)), 4)


def _track_identity_risk_reasons(
    track_item: Mapping[str, Any],
    label_context: Mapping[str, Any],
    identity_confidence: float,
) -> list[str]:
    reasons = []
    same_class_track_count = int(label_context.get("same_class_track_count") or 0)
    max_same_class_per_frame = int(label_context.get("max_same_class_per_frame") or 0)
    if same_class_track_count > 1:
        reasons.append(f"same_class_track_count={same_class_track_count}")
    if max_same_class_per_frame > 1:
        reasons.append(f"max_same_class_per_frame={max_same_class_per_frame}")
    if not _explicit_track_identity(track_item) and same_class_track_count > 1:
        reasons.append("inferred track without model track_id in same-class scene")
    if identity_confidence < 0.65:
        reasons.append(f"identity_confidence_below_threshold={identity_confidence:.3f}")
    return reasons


def _average_detection_confidence(track: list[dict[str, Any]]) -> float | None:
    values = [_float(item.get("confidence"), None) for item in track if item.get("confidence") is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _bbox_scale(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(abs(x2 - x1), abs(y2 - y1), 1.0)


def _image_pair_evidence(
    session: Path,
    micro: Mapping[str, Any],
    assets: list[Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    interaction = _as_dict(micro.get("interaction"))
    action_type = str(_as_dict(micro.get("text_description")).get("action_type") or "")
    primary = str(interaction.get("primary_object") or "")
    keyframes = _as_dict(micro.get("keyframes"))
    contact = _with_frame_role(_image_metrics(_resolve_path(session, keyframes.get("contact_frame"))), "contact")
    peak = _with_frame_role(_image_metrics(_resolve_path(session, keyframes.get("peak_frame"))), "peak")
    release = _with_frame_role(_image_metrics(_resolve_path(session, keyframes.get("release_frame"))), "release")
    metrics = [item for item in (contact, peak, release) if item]
    quality_limitations = _image_metric_limitations(metrics)
    liquid_model = _capability_summary(inventory, "liquid_stream_segmentation")
    cap_lid_model = _capability_summary(inventory, "cap_lid_detection")
    events = []
    if len(metrics) >= 2:
        color_delta = _color_delta(metrics[0], metrics[-1])
        level_values = [item["liquid_level_y_norm"] for item in metrics if item.get("liquid_level_y_norm") is not None]
        if color_delta >= 20.0 and _containerish(primary):
            events.append(
                _event(
                    micro,
                    evidence_type="container_color_change",
                    object_label=primary,
                    visual_confirmation_level="classical_image_change_detected",
                    confidence=min(0.82, 0.45 + color_delta / 100.0),
                    evidence_reasons=[f"keyframe color histogram changed by {color_delta:.2f}"],
                    limitations=["classical image metric; not semantic color label confirmation", *quality_limitations],
                    metrics={
                        "color_delta": round(color_delta, 4),
                        "color_change_indicator": _color_change_indicator(metrics[0], metrics[-1], color_delta),
                        "image_metrics": metrics,
                        "quality_limitations": quality_limitations,
                    },
                    asset_refs=_asset_refs(assets),
                    payload={},
                )
            )
        if len(level_values) >= 2:
            level_delta = abs(level_values[-1] - level_values[0])
            if level_delta >= 0.08 and _liquid_related(primary, action_type):
                level = "liquid_level_change_confirmed" if liquid_model["available"] else "liquid_level_change_candidate"
                base_confidence = 0.58 + min(level_delta, 0.32)
                confidence = min(0.9 if liquid_model["available"] else 0.74, base_confidence + (0.08 if liquid_model["available"] else 0.0))
                limitations = (
                    ["segmentation/meniscus capability is present in model_inventory; volume calibration still depends on container ROI"]
                    if liquid_model["available"]
                    else ["liquid stream segmentation unavailable in model_inventory; level estimate uses uncalibrated horizontal-edge heuristic"]
                )
                events.append(
                    _event(
                        micro,
                        evidence_type="liquid_level_change",
                        object_label=primary,
                        visual_confirmation_level=level,
                        confidence=confidence,
                        evidence_reasons=[f"horizontal level estimate changed by {level_delta:.3f}"],
                        limitations=[*limitations, *quality_limitations],
                        metrics={
                            "liquid_level_delta": round(level_delta, 4),
                            "level_values": [round(float(value), 4) for value in level_values],
                            "level_indicators": [item.get("liquid_level_indicator", {}) for item in metrics],
                            "model_capability": liquid_model,
                            "image_metrics": metrics,
                            "quality_limitations": quality_limitations,
                        },
                        asset_refs=_asset_refs(assets),
                        payload={},
                    )
                )
    if _liquid_related(primary, action_type):
        flow_confirmed = bool(liquid_model["available"] and metrics)
        flow_level = "liquid_flow_confirmed" if flow_confirmed else "liquid_flow_candidate"
        flow_limitations = (
            ["liquid stream/meniscus capability available in model_inventory; attach mask outputs for calibrated flow direction and volume"]
            if flow_confirmed
            else ["does not confirm visible liquid stream without a trained fluid/level model"]
        )
        events.append(
            _event(
                micro,
                evidence_type="liquid_flow_candidate_visual",
                object_label=primary,
                visual_confirmation_level=flow_level,
                confidence=0.74 if flow_confirmed else (0.52 if metrics else 0.4),
                evidence_reasons=[
                    "pipetting/liquid-transfer context detected",
                    f"analyzable_keyframes={len(metrics)}",
                    f"liquid_stream_segmentation_available={str(liquid_model['available']).lower()}",
                ],
                limitations=[*flow_limitations, *quality_limitations],
                metrics={
                    "analyzable_keyframe_count": len(metrics),
                    "model_capability": liquid_model,
                    "keyframe_visual_indicators": metrics,
                    "quality_limitations": quality_limitations,
                },
                asset_refs=_asset_refs(assets),
                payload={},
            )
        )
    if _containerish(primary):
        cap_tokens = _object_tokens(micro)
        cap_lid_tokens = sorted(token for token in cap_tokens if _is_cap_lid_token(token))
        if cap_lid_tokens and cap_lid_model["available"]:
            level = "container_open_close_confirmed"
            confidence = 0.8
            reasons = ["cap/lid detection capability available in model_inventory", "cap/lid object token detected in visual metadata"]
            limitations = ["open-vs-close direction still depends on before/after cap position evidence"]
        elif cap_lid_tokens:
            level = "container_open_close_candidate"
            confidence = 0.62
            reasons = ["cap/lid object token detected, but cap/lid detector capability is unavailable in model_inventory"]
            limitations = ["cap/lid state classifier unavailable; open/close state needs detector evidence or manual confirmation"]
        else:
            level = "container_open_close_candidate"
            confidence = 0.42
            reasons = ["container interaction present but no cap/lid detector evidence"]
            limitations = ["open/close state requires cap/lid detector or manual confirmation"]
        events.append(
            _event(
                micro,
                evidence_type="container_open_close",
                object_label=primary,
                visual_confirmation_level=level,
                confidence=confidence,
                evidence_reasons=reasons,
                limitations=[*limitations, *quality_limitations],
                metrics={
                    "object_tokens": sorted(cap_tokens)[:20],
                    "cap_lid_tokens": cap_lid_tokens[:20],
                    "model_capability": cap_lid_model,
                    "container_state_indicators": _container_state_indicators(metrics, cap_lid_tokens, cap_lid_model),
                    "quality_limitations": quality_limitations,
                },
                asset_refs=_asset_refs(assets),
                payload={},
            )
        )
    return events


def _panel_evidence(session: Path, micro: Mapping[str, Any], assets: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    interaction = _as_dict(micro.get("interaction"))
    text = str(_as_dict(micro.get("text_description")).get("index_text") or _as_dict(micro.get("text_description")).get("summary") or "")
    primary = str(interaction.get("primary_object") or "")
    if not _panel_related(primary, text):
        return []
    keyframes = _as_dict(micro.get("keyframes"))
    image_path = _resolve_path(session, keyframes.get("peak_frame") or keyframes.get("contact_frame"))
    ocr_text, ocr_reason = _optional_ocr(image_path)
    if ocr_text:
        return [
            _event(
                micro,
                evidence_type="equipment_panel_ocr",
                object_label=primary,
                visual_confirmation_level="ocr_text_detected",
                confidence=0.8,
                evidence_reasons=[ocr_reason, f"ocr_text={ocr_text[:80]}"],
                limitations=["OCR quality depends on panel crop and installed OCR engine"],
                metrics={"ocr_text": ocr_text},
                asset_refs=_asset_refs(assets),
                payload={},
            )
        ]
    contrast = _image_metrics(image_path)
    confidence = 0.55 if contrast else 0.38
    return [
        _event(
            micro,
            evidence_type="equipment_control_change",
            object_label=primary,
            visual_confirmation_level="candidate_requires_panel_ocr_or_control_detector",
            confidence=confidence,
            evidence_reasons=["equipment/readout context detected", ocr_reason],
            limitations=["no OCR text confirmed; button/knob state requires panel detector"],
            metrics={"image_metrics": contrast or {}},
            asset_refs=_asset_refs(assets),
            payload={},
        )
    ]


def _model_observation_evidence(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for row in rows:
        observation_id = str(row.get("observation_id") or "")
        evidence_type = _model_observation_evidence_type(row)
        if not evidence_type:
            continue
        object_label = str(row.get("object_label") or "object")
        visual_level = _model_observation_visual_level(row, evidence_type)
        confidence = _float(row.get("confidence"), 0.75)
        confirmation_level = str(row.get("confirmation_level") or _confirmation_level_for_event(evidence_type, visual_level))
        reasons = _dedupe(
            [
                *_strings(row.get("evidence_reasons")),
                f"model_observation_event={observation_id}",
                f"confirmation_level={confirmation_level or 'unknown'}",
            ]
        )
        refs = row.get("asset_refs") or []
        evidence.append(
            {
                "evidence_id": f"{observation_id}:{evidence_type}",
                "session_id": row.get("session_id"),
                "evidence_type": evidence_type,
                "segment_id": row.get("segment_id"),
                "micro_segment_id": row.get("micro_segment_id"),
                "global_start_time": row.get("global_start_time"),
                "global_end_time": row.get("global_end_time"),
                "object_label": object_label,
                "action_type": _model_observation_action_type(row),
                "visual_confirmation_level": visual_level,
                "confirmation_level": confirmation_level,
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                "confidence_reasons": reasons,
                "evidence_reasons": reasons,
                "limitations": _strings(row.get("limitations")),
                "multiview_consistency": {
                    "status": "model_observation_single_view" if row.get("view") else "not_available",
                    "consistent": None,
                    "views": [str(row.get("view"))] if row.get("view") else [],
                    "supporting_frame_counts": {str(row.get("view")): 1} if row.get("view") else {},
                    "object_label": object_label,
                },
                "metrics": {
                    "model_observation_id": observation_id,
                    "source_type": row.get("source_type"),
                    "observation_type": row.get("observation_type"),
                    "event_type": row.get("event_type"),
                    "confirmation_level": confirmation_level,
                    "state": row.get("state"),
                    "measurement": row.get("measurement") or {},
                    "model_metrics": row.get("metrics") or {},
                },
                "asset_refs": refs,
                "evidence_refs": refs,
                "requires_human_confirmation": not _confirmed_or_measured_level(visual_level, confirmation_level) or confidence < 0.5,
                "payload": {"source": "model_observation_events", "model_observation": dict(row)},
            }
        )
    return evidence


def _model_observation_evidence_type(row: Mapping[str, Any]) -> str:
    event_type = str(row.get("event_type") or "")
    observation_type = str(row.get("observation_type") or row.get("source_type") or "")
    if event_type == "liquid_level_measured":
        return "liquid_level_change"
    if event_type.startswith("container_color_change"):
        return "container_color_change"
    if event_type == "liquid_flow_observed" or observation_type == "liquid_segmentation":
        return "liquid_flow_candidate_visual"
    if event_type.startswith("equipment_panel_state") or event_type.startswith("equipment_control_state") or observation_type == "equipment_panel_state":
        return "equipment_control_change"
    if event_type == "container_state_confirmed" or event_type.startswith("container_open") or event_type.startswith("container_close") or observation_type == "container_state":
        return "container_open_close"
    if event_type in {"object_movement_measured", "object_track_measured"}:
        return "object_trajectory_movement"
    if event_type == "object_track_observed" or observation_type == "object_track":
        return "object_track_observation"
    return ""


def _model_observation_visual_level(row: Mapping[str, Any], evidence_type: str) -> str:
    confirmation_level = str(row.get("confirmation_level") or "").lower()
    event_type = str(row.get("event_type") or "")
    if evidence_type == "liquid_level_change":
        return "liquid_level_measured" if confirmation_level == "measured" else "liquid_level_change_confirmed"
    if evidence_type == "liquid_flow_candidate_visual":
        return "liquid_flow_confirmed" if confirmation_level in {"confirmed", "measured"} else "liquid_flow_candidate"
    if evidence_type == "equipment_control_change":
        if confirmation_level == "measured":
            return "equipment_panel_state_measured"
        return "equipment_panel_state_confirmed" if confirmation_level == "confirmed" else "equipment_panel_state_candidate"
    if evidence_type == "container_open_close":
        return "container_state_confirmed" if confirmation_level in {"confirmed", "measured"} else "container_open_close_candidate"
    if evidence_type == "container_color_change":
        if confirmation_level == "measured":
            return "container_color_change_measured"
        return "container_color_change_confirmed" if confirmation_level == "confirmed" else "container_color_change_candidate"
    if evidence_type == "object_trajectory_movement":
        return "trajectory_measured" if confirmation_level == "measured" else "trajectory_confirmed"
    if evidence_type == "object_track_observation":
        return "track_observed_confirmed" if confirmation_level in {"confirmed", "measured"} else "track_observed_candidate"
    if event_type:
        return f"{event_type}_{confirmation_level or 'observed'}"
    return confirmation_level or "model_observation_confirmed"


def _model_observation_action_type(row: Mapping[str, Any]) -> Any:
    payload = _as_dict(row.get("payload"))
    source_row = _as_dict(payload.get("source_row"))
    return _first_non_empty(row.get("action_type"), source_row.get("action_type"), source_row.get("action"))


def _prefer_model_observation_upgrades(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    upgraded_keys = {
        (str(row.get("micro_segment_id") or ""), str(row.get("evidence_type") or ""))
        for row in rows
        if _is_model_observation_evidence(row)
        and _confirmed_or_measured_level(str(row.get("visual_confirmation_level") or ""), str(_as_dict(row.get("metrics")).get("confirmation_level") or ""))
    }
    if not upgraded_keys:
        return rows
    output = []
    for row in rows:
        key = (str(row.get("micro_segment_id") or ""), str(row.get("evidence_type") or ""))
        if not _is_model_observation_evidence(row) and key in upgraded_keys and _candidate_visual_level(str(row.get("visual_confirmation_level") or "")):
            continue
        output.append(row)
    return output


def _is_model_observation_evidence(row: Mapping[str, Any]) -> bool:
    return str(_as_dict(row.get("payload")).get("source") or "") == "model_observation_events"


def _confirmed_or_measured_level(visual_level: str, confirmation_level: str = "") -> bool:
    text = f"{visual_level} {confirmation_level}".lower()
    return "confirmed" in text or "measured" in text


def _confirmation_level_for_event(evidence_type: str, visual_level: str) -> str:
    text = str(visual_level or "").lower()
    if _candidate_visual_level(text) or "candidate" in text or "risk" in text or "requires" in text:
        return "candidate"
    if evidence_type in {"object_trajectory_movement", "liquid_level_change"}:
        return "measured"
    if "measured" in text or "ocr_text_detected" in text:
        return "measured"
    if "confirmed" in text or "detected" in text or "observed" in text:
        return "confirmed"
    return "candidate"


def _multiview_consistency(rows: Iterable[Mapping[str, Any]], object_label: str = "") -> dict[str, Any]:
    per_view: dict[str, int] = {}
    for row in rows:
        view = str(row.get("view") or row.get("source_view") or row.get("camera") or "unknown")
        if not view:
            view = "unknown"
        if object_label:
            labels = {
                str(det.get("label") or det.get("object_label") or "")
                for det in _detections(row)
                if isinstance(det, Mapping)
            }
            interactions = {
                str(item.get("object_label") or "")
                for item in row.get("hand_object_interactions") or []
                if isinstance(item, Mapping)
            }
            if object_label not in labels and object_label not in interactions:
                continue
        per_view[view] = per_view.get(view, 0) + 1
    views = sorted(per_view)
    if len(views) >= 2:
        status = "consistent_multiview"
        consistent: bool | None = True
    elif len(views) == 1:
        status = "single_view_only"
        consistent = None
    else:
        status = "not_available"
        consistent = None
    return {
        "status": status,
        "consistent": consistent,
        "views": views,
        "supporting_frame_counts": per_view,
        "object_label": object_label or None,
    }


def _row_evidence_refs(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for row in rows:
        refs.append(
            {
                "asset_type": "yolo_frame_row",
                "source_view": row.get("source_view") or row.get("view"),
                "frame_index": row.get("frame_index"),
                "sample_index": row.get("sample_index"),
                "local_time_sec": row.get("local_time_sec") or row.get("time_sec"),
                "global_time": row.get("global_time"),
                "video_path": row.get("video_path"),
            }
        )
    return refs


def _event(
    micro: Mapping[str, Any],
    *,
    evidence_type: str,
    object_label: str,
    visual_confirmation_level: str,
    confidence: float,
    evidence_reasons: list[str],
    limitations: list[str],
    metrics: dict[str, Any],
    asset_refs: list[dict[str, Any]],
    payload: dict[str, Any],
    multiview_consistency: dict[str, Any] | None = None,
    evidence_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    micro_id = str(micro.get("micro_segment_id") or "")
    bounded_confidence = round(max(0.0, min(1.0, confidence)), 4)
    confirmation_level = _confirmation_level_for_event(evidence_type, visual_confirmation_level)
    refs = evidence_refs if evidence_refs is not None else asset_refs
    multiview = multiview_consistency or {"status": "not_available", "consistent": None, "views": [], "supporting_frame_counts": {}}
    return {
        "evidence_id": f"{micro_id}:{evidence_type}:{object_label or 'object'}",
        "session_id": micro.get("session_id"),
        "evidence_type": evidence_type,
        "segment_id": micro.get("parent_segment_id") or micro.get("segment_id"),
        "micro_segment_id": micro_id,
        "global_start_time": micro.get("global_start_time"),
        "global_end_time": micro.get("global_end_time"),
        "object_label": object_label,
        "action_type": _as_dict(micro.get("text_description")).get("action_type"),
        "visual_confirmation_level": visual_confirmation_level,
        "confirmation_level": confirmation_level,
        "confidence": bounded_confidence,
        "confidence_reasons": evidence_reasons,
        "evidence_reasons": evidence_reasons,
        "limitations": limitations,
        "multiview_consistency": multiview,
        "metrics": metrics,
        "asset_refs": asset_refs,
        "evidence_refs": refs,
        "requires_human_confirmation": confirmation_level == "candidate" or bounded_confidence < 0.65,
        "payload": {**payload, "source": "advanced_vision_evidence"},
    }


def _image_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            original_width, original_height = image.size
            rgb = image.convert("RGB").resize((160, 120))
            stat = ImageStat.Stat(rgb)
            gray = rgb.convert("L")
            pixels = list(gray.tobytes())
            rows = []
            for y in range(1, gray.height):
                current_start = y * gray.width
                previous_start = (y - 1) * gray.width
                current = sum(pixels[current_start : current_start + gray.width]) / gray.width
                previous = sum(pixels[previous_start : previous_start + gray.width]) / gray.width
                rows.append(abs(current - previous))
            if rows:
                strongest = max(range(len(rows)), key=lambda idx: rows[idx])
                level_y = (strongest + 1) / gray.height if rows[strongest] > 8 else None
                edge_strength = max(rows)
            else:
                level_y = None
                edge_strength = 0.0
            horizontal_diffs = []
            vertical_diffs = []
            for y in range(gray.height):
                row_start = y * gray.width
                for x in range(1, gray.width):
                    horizontal_diffs.append(abs(pixels[row_start + x] - pixels[row_start + x - 1]))
            for y in range(1, gray.height):
                row_start = y * gray.width
                previous_start = (y - 1) * gray.width
                for x in range(gray.width):
                    vertical_diffs.append(abs(pixels[row_start + x] - pixels[previous_start + x]))
            sharpness = max(_mean(horizontal_diffs + vertical_diffs), edge_strength)
            contrast = _mean(stat.stddev)
            brightness = _mean(stat.mean)
            saturation = (max(stat.mean) - min(stat.mean)) / 255.0 if stat.mean else 0.0
            edge_noise = median(rows) if rows else 0.0
            edge_to_noise = edge_strength / max(edge_noise, 1.0)
            level_confidence = 0.0
            if level_y is not None:
                level_confidence = min(0.95, 0.35 + min(edge_strength, 40.0) / 80.0 + min(edge_to_noise, 5.0) / 20.0)
            quality_limitations = _frame_quality_limitations(original_width, original_height, brightness, contrast, sharpness)
            level_limitations = []
            if level_y is None:
                level_limitations.append("no strong horizontal liquid-level edge detected")
            if edge_to_noise < 1.4:
                level_limitations.append("horizontal edge is not well separated from frame texture/noise")
            return {
                "path": str(path),
                "image_size": {"width": original_width, "height": original_height},
                "analysis_size": {"width": rgb.width, "height": rgb.height},
                "mean_rgb": [round(value, 3) for value in stat.mean],
                "stddev_rgb": [round(value, 3) for value in stat.stddev],
                "liquid_level_y_norm": level_y,
                "horizontal_edge_strength": round(edge_strength, 4),
                "color_profile": {
                    "mean_rgb": [round(value, 3) for value in stat.mean],
                    "stddev_rgb": [round(value, 3) for value in stat.stddev],
                    "dominant_color_family": _dominant_color_family(stat.mean),
                    "brightness": round(brightness, 4),
                    "saturation_estimate": round(saturation, 4),
                    "colorfulness": round(_mean(stat.stddev), 4),
                },
                "liquid_level_indicator": {
                    "status": "detected" if level_y is not None else "not_detected",
                    "y_norm": round(level_y, 4) if level_y is not None else None,
                    "edge_strength": round(edge_strength, 4),
                    "edge_to_noise_ratio": round(edge_to_noise, 4),
                    "confidence": round(level_confidence, 4),
                    "quality_limitations": level_limitations,
                },
                "container_state_indicators": {
                    "strong_horizontal_edge_count": sum(1 for value in rows if value > 8),
                    "supports_liquid_level_estimate": level_y is not None,
                    "supports_color_comparison": contrast >= 3.0 and 15.0 <= brightness <= 240.0,
                },
                "frame_quality": {
                    "status": "usable" if not quality_limitations else "limited",
                    "brightness": round(brightness, 4),
                    "contrast_score": round(contrast, 4),
                    "sharpness_score": round(sharpness, 4),
                    "limitations": quality_limitations,
                },
            }
    except OSError:
        return None


def _with_frame_role(metrics: dict[str, Any] | None, role: str) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {**metrics, "frame_role": role}


def _image_metric_limitations(metrics: list[Mapping[str, Any]]) -> list[str]:
    limitations = []
    for item in metrics:
        role = str(item.get("frame_role") or "keyframe")
        frame_quality = _as_dict(item.get("frame_quality"))
        for limitation in frame_quality.get("limitations") or []:
            limitations.append(f"{role}:{limitation}")
        level_indicator = _as_dict(item.get("liquid_level_indicator"))
        for limitation in level_indicator.get("quality_limitations") or []:
            if "not_detected" in str(level_indicator.get("status") or ""):
                limitations.append(f"{role}:{limitation}")
    return _dedupe(limitations)[:8]


def _capability_summary(inventory: Mapping[str, Any], name: str) -> dict[str, Any]:
    capabilities = _as_dict(inventory.get("capabilities"))
    capability = _as_dict(capabilities.get(name))
    available = capability.get("available")
    if isinstance(available, str):
        is_available = available.strip().lower() in {"1", "true", "yes", "available"}
    else:
        is_available = bool(available)
    return {
        "name": name,
        "available": is_available,
        "classes": [str(item) for item in list(capability.get("classes") or [])],
    }


def _color_change_indicator(first: Mapping[str, Any], last: Mapping[str, Any], color_delta: float) -> dict[str, Any]:
    first_profile = _as_dict(first.get("color_profile"))
    last_profile = _as_dict(last.get("color_profile"))
    return {
        "delta": round(color_delta, 4),
        "first_frame_role": first.get("frame_role"),
        "last_frame_role": last.get("frame_role"),
        "first_color_family": first_profile.get("dominant_color_family"),
        "last_color_family": last_profile.get("dominant_color_family"),
        "brightness_delta": round(abs(_float(last_profile.get("brightness"), 0.0) - _float(first_profile.get("brightness"), 0.0)), 4),
        "saturation_delta": round(abs(_float(last_profile.get("saturation_estimate"), 0.0) - _float(first_profile.get("saturation_estimate"), 0.0)), 4),
    }


def _container_state_indicators(
    image_metrics: list[Mapping[str, Any]],
    cap_lid_tokens: list[str],
    cap_lid_model: Mapping[str, Any],
) -> dict[str, Any]:
    frame_support = []
    for item in image_metrics:
        indicators = _as_dict(item.get("container_state_indicators"))
        frame_support.append(
            {
                "frame_role": item.get("frame_role"),
                "supports_color_comparison": bool(indicators.get("supports_color_comparison")),
                "supports_liquid_level_estimate": bool(indicators.get("supports_liquid_level_estimate")),
                "strong_horizontal_edge_count": int(indicators.get("strong_horizontal_edge_count") or 0),
            }
        )
    return {
        "cap_lid_detection_available": bool(cap_lid_model.get("available")),
        "cap_lid_model_classes": list(cap_lid_model.get("classes") or []),
        "cap_lid_token_count": len(cap_lid_tokens),
        "cap_lid_tokens": cap_lid_tokens,
        "state_signal": "cap_lid_detected" if cap_lid_tokens else "container_interaction_only",
        "keyframe_support": frame_support,
    }


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _frame_quality_limitations(width: int, height: int, brightness: float, contrast: float, sharpness: float) -> list[str]:
    limitations = []
    if width < 40 or height < 40:
        limitations.append("image too small for reliable state metrics")
    if brightness < 18.0:
        limitations.append("image is underexposed")
    elif brightness > 238.0:
        limitations.append("image is overexposed")
    if contrast < 3.0:
        limitations.append("low contrast limits color/state estimates")
    if sharpness < 2.0:
        limitations.append("low edge detail limits liquid level/state estimates")
    return limitations


def _dominant_color_family(mean_rgb: Iterable[float]) -> str:
    values = list(mean_rgb)
    if len(values) < 3:
        return "unknown"
    red, green, blue = [float(value) for value in values[:3]]
    brightness = (red + green + blue) / 3.0
    spread = max(red, green, blue) - min(red, green, blue)
    if brightness < 35:
        return "dark"
    if brightness > 225 and spread < 25:
        return "white"
    if spread < 18:
        return "gray"
    if red >= green and red >= blue:
        return "yellow_or_orange" if green > blue + 25 else "red_or_magenta"
    if green >= red and green >= blue:
        return "green_or_cyan" if blue > red + 20 else "green"
    return "blue_or_purple"


def _optional_ocr(path: Path | None) -> tuple[str, str]:
    if path is None or not path.exists():
        return "", "no panel image available"
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return "", "pytesseract not installed"
    try:
        text = pytesseract.image_to_string(Image.open(path), config="--psm 6").strip()
    except Exception as exc:
        return "", f"ocr_failed={exc}"
    return text, "pytesseract OCR executed"


def _load_yolo_rows(session: Path) -> list[dict[str, Any]]:
    candidates = [
        session / "cv_outputs" / "yolo_micro_frame_rows.jsonl",
        session / "cv_outputs" / "yolo_frame_rows.jsonl",
        session / "metadata" / "yolo_frame_rows.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return read_jsonl(path)
    return []


def _rows_for_micro(rows: list[Mapping[str, Any]], micro: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    start = _float(micro.get("start_sec"), None)
    end = _float(micro.get("end_sec"), None)
    if start is None or end is None:
        return rows
    scoped = []
    for row in rows:
        time_value = _row_time(row)
        if start <= time_value <= end:
            scoped.append(row)
    return scoped


def _detections(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    detections = row.get("detections")
    if isinstance(detections, list):
        return [det for det in detections if isinstance(det, Mapping)]
    interactions = row.get("hand_object_interactions")
    output = []
    if isinstance(interactions, list):
        for item in interactions:
            if isinstance(item, Mapping):
                output.append(
                    {
                        "label": item.get("object_label") or item.get("primary_object"),
                        "bbox": item.get("object_bbox"),
                        "confidence": item.get("score"),
                        "track_id": item.get("object_track_id") or item.get("track_id"),
                        "track_source": item.get("track_source"),
                    }
                )
    return output


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return float(value[0]), float(value[1]), float(value[2]), float(value[3])
    except (TypeError, ValueError):
        return None


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _row_time(row: Mapping[str, Any]) -> float:
    return _float(row.get("alignment_time_sec"), _float(row.get("session_time_sec"), _float(row.get("local_time_sec"), _float(row.get("time_sec"), 0.0))))


def _assets_by_micro(rows: list[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        micro_id = str(row.get("micro_segment_id") or row.get("source_id") or "")
        if micro_id:
            result[micro_id].append(row)
    return result


def _asset_refs(assets: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs = []
    for asset in assets:
        refs.append(
            {
                "asset_id": asset.get("asset_id"),
                "asset_type": asset.get("asset_type"),
                "source_type": asset.get("source_type"),
                "path": asset.get("path"),
                "quality": asset.get("quality"),
            }
        )
    return refs


def _resolve_path(session: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    parts = path.parts
    for index, part in enumerate(parts):
        if part == session.name and index < len(parts) - 1:
            return session / Path(*parts[index + 1 :])
    if path.exists():
        return path
    return session / path


def _color_delta(first: Mapping[str, Any], last: Mapping[str, Any]) -> float:
    a = first.get("mean_rgb") or []
    b = last.get("mean_rgb") or []
    if len(a) < 3 or len(b) < 3:
        return 0.0
    return math.sqrt(sum((float(a[idx]) - float(b[idx])) ** 2 for idx in range(3)))


def _containerish(value: str) -> bool:
    text = str(value or "").lower()
    return any(token in text for token in CONTAINER_TOKENS)


def _liquid_related(primary: str, action_type: str) -> bool:
    text = f"{primary} {action_type}".lower()
    return any(token in text for token in LIQUID_TOKENS)


def _panel_related(primary: str, text: str) -> bool:
    haystack = f"{primary} {text}".lower()
    return any(token in haystack for token in PANEL_TOKENS)


def _object_tokens(micro: Mapping[str, Any]) -> set[str]:
    interaction = _as_dict(micro.get("interaction"))
    values = [interaction.get("primary_object"), interaction.get("secondary_objects"), interaction.get("detected_objects"), interaction.get("interaction_type")]
    tokens = set()
    for value in values:
        if isinstance(value, list):
            for item in value:
                tokens.update(_normalized_tokens(str(item)))
        elif value:
            tokens.update(_normalized_tokens(str(value)))
    return {token for token in tokens if token}


def _normalized_tokens(value: str) -> set[str]:
    raw = str(value or "").strip().lower()
    compound = raw.replace("-", "_").replace(" ", "_")
    while "__" in compound:
        compound = compound.replace("__", "_")
    return {compound, *compound.split("_")}


def _is_cap_lid_token(value: Any) -> bool:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return any(str(token).replace("-", "_") in text for token in CAP_LID_TOKENS)


def _load_or_discover_inventory(session: Path) -> dict[str, Any]:
    existing = session / "metadata" / "model_inventory.json"
    if existing.exists():
        try:
            return load_model_inventory(existing)
        except Exception:
            pass
    try:
        return discover_lab_assets()
    except Exception:
        return {}


def _stable(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in EVIDENCE_FIELDS}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float(value: Any, default: float | None = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0 if default is None else default


def _first_text(values: Iterable[Any]) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def _manifest_session_id(session: Path) -> str:
    manifest = session / "manifest.json"
    if not manifest.exists():
        return ""
    try:
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("session_id") or "")


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value) != "":
            return value
    return None


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)] if str(value) else []


def _dedupe(values: Iterable[Any]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _candidate_visual_level(value: str) -> bool:
    text = str(value or "").lower()
    return text.startswith("candidate") or text.endswith("_candidate") or "_candidate_" in text


__all__ = ["build_advanced_vision_evidence", "load_advanced_vision_evidence"]

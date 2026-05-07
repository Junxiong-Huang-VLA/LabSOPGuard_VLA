from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from .protocol_graph import ProtocolGraphBuilder
from .schemas import METADATA_VERSION, ProtocolStepNode, StepCandidate, StepEvidenceBundle, StepPromotionDecision, write_json

GRADE_SCORE = {"strong": 0.9, "medium": 0.65, "weak": 0.35}
STATUS_PENALTY = {"auto_confirmed": 0.0, "candidate_review": 0.12, "low_confidence": 0.28}


class StepBridgeEngine:
    def __init__(self) -> None:
        self.graph_builder = ProtocolGraphBuilder()

    def run(
        self,
        *,
        experiment_id: str,
        output_dir: str | Path,
        steps: List[Dict[str, Any]],
        physical_events_payload: Dict[str, Any],
        preprocessing_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        output_path = Path(output_dir)
        events = physical_events_payload.get("events") if isinstance(physical_events_payload, dict) else physical_events_payload
        events = events or []
        tracked_objects = (physical_events_payload or {}).get("tracked_objects") or (preprocessing_payload or {}).get("tracked_objects") or []
        tracked_by_id = {str(item.get("track_id")): item for item in tracked_objects if item.get("track_id")}
        protocol_nodes = self.graph_builder.build(steps, protocol_payload={"output_dir": str(output_path)})

        candidates: List[Dict[str, Any]] = []
        bundles: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        missing_steps: List[Dict[str, Any]] = []
        blocked_steps: List[Dict[str, Any]] = []
        assigned_event_ids: set[str] = set()

        previous_confirmed_index = -1
        node_results: Dict[str, Dict[str, Any]] = {}
        for node in protocol_nodes:
            matched = self._match_events(node, events, assigned_event_ids)
            bundle = self._build_bundle(node, matched, tracked_by_id)
            decision = self._decide(node, matched, bundle, previous_confirmed_index)
            candidate = self._candidate(experiment_id, node, matched, decision, bundle)
            if node.event_reuse_policy != "allow_reuse" and candidate.candidate_status != "inferred":
                assigned_event_ids.update(str(event.get("event_id")) for event in matched if event.get("event_id"))
            if decision.decision == "promote_confirmed":
                previous_confirmed_index = max(previous_confirmed_index, node.step_index)
            elif decision.decision == "hold_for_review" and "predecessor_not_confirmed" in decision.blocking_issues:
                blocked_steps.append({"protocol_step_id": node.protocol_step_id, "reason": "predecessor_not_confirmed"})
            if not matched or decision.decision in {"mark_inferred", "hold_for_review"} and "missing_required_event" in decision.blocking_issues:
                missing_steps.append({"protocol_step_id": node.protocol_step_id, "required_event_types": node.required_event_types})
            candidates.append(candidate.to_dict())
            bundles.append(bundle.to_dict())
            decisions.append(decision.to_dict())
            node_results[node.protocol_step_id] = {"candidate": candidate.to_dict(), "decision": decision.to_dict(), "bundle": bundle.to_dict()}

        out_of_order_steps = self._out_of_order(protocol_nodes, candidates, events)
        summary = self._summary(experiment_id, protocol_nodes, candidates, decisions, missing_steps, out_of_order_steps, blocked_steps)
        payload = {
            "schema_version": "step_candidates.v1",
            "metadata_version": METADATA_VERSION,
            "experiment_id": experiment_id,
            "protocol_graph": [node.to_dict() for node in protocol_nodes],
            "step_candidates": candidates,
            "evidence_bundles": bundles,
            "promotion_decisions": decisions,
        }
        write_json(output_path / "step_candidates.json", payload)
        write_json(output_path / "step_bridge_summary.json", summary)
        return {"step_candidates": payload, "step_bridge_summary": summary}

    def _match_events(self, node: ProtocolStepNode, events: List[Dict[str, Any]], assigned_event_ids: set[str]) -> List[Dict[str, Any]]:
        allowed = set(node.required_event_types + node.optional_event_types)
        if not allowed:
            return []
        scored: List[tuple[float, Dict[str, Any]]] = []
        for event_rank, event in enumerate(sorted(events, key=lambda item: float(item.get("start_time_sec") or 0.0))):
            event_type = event.get("event_type")
            if event_type not in allowed:
                continue
            score = self._assignment_score(node, event, event_rank, max(1, len(events)), str(event.get("event_id")) in assigned_event_ids)
            if node.event_reuse_policy == "unique" and str(event.get("event_id")) in assigned_event_ids:
                continue
            if node.event_reuse_policy == "prefer_unique" and str(event.get("event_id")) in assigned_event_ids and score < 0.72:
                continue
            scored.append((score, event))
        required_types = set(node.required_event_types)
        required = [(score, event) for score, event in scored if event.get("event_type") in required_types]
        optional = [(score, event) for score, event in scored if event.get("event_type") not in required_types and event.get("evidence_grade") != "weak"]
        required.sort(key=lambda item: (-item[0], float(item[1].get("start_time_sec") or 0.0)))
        optional.sort(key=lambda item: (-item[0], float(item[1].get("start_time_sec") or 0.0)))
        selected = [event for _, event in required[:3]]
        selected.extend(event for _, event in optional[: max(0, 6 - len(selected))])
        selected.sort(key=lambda event: float(event.get("start_time_sec") or 0.0))
        return selected

    @staticmethod
    def _assignment_score(node: ProtocolStepNode, event: Dict[str, Any], event_rank: int, event_count: int, reused: bool) -> float:
        event_type = event.get("event_type")
        score = 0.25
        if event_type in set(node.required_event_types):
            score += 0.35
        elif event_type in set(node.optional_event_types):
            score += 0.16
        score += GRADE_SCORE.get(str(event.get("evidence_grade")), 0.35) * 0.18
        score -= STATUS_PENALTY.get(str(event.get("review_status")), 0.15) * 0.35
        if event.get("direction_status") == "confirmed":
            score += 0.08
        elif event.get("direction_status") == "unknown":
            score -= 0.08
        if event.get("state_confidence") is not None:
            score += min(0.08, max(0.0, float(event.get("state_confidence") or 0.0) * 0.08))
        tq = event.get("track_quality_summary") or {}
        score += min(0.08, float(tq.get("avg_track_confidence") or event.get("track_confidence") or 0.0) * 0.08)
        expected_rank = node.step_index / max(1, event_count)
        observed_rank = event_rank / max(1, event_count)
        score -= min(0.12, abs(expected_rank - observed_rank) * 0.18)
        if reused:
            score -= 0.25
        return max(0.0, min(1.0, score))

    def _build_bundle(self, node: ProtocolStepNode, events: List[Dict[str, Any]], tracked_by_id: Dict[str, Dict[str, Any]]) -> StepEvidenceBundle:
        linked_paths: List[str] = []
        direction_signals: List[Dict[str, Any]] = []
        state_signals: List[Dict[str, Any]] = []
        track_ids: List[str] = []
        event_summaries: List[str] = []
        for event in events:
            asset = event.get("asset_pack") or {}
            for key in ("clip_path", "preview_path"):
                if asset.get(key):
                    linked_paths.append(str(asset[key]))
            linked_paths.extend(str(path) for path in asset.get("keyframe_paths") or [])
            if event.get("direction_status") is not None:
                direction_signals.append({
                    "event_id": event.get("event_id"),
                    "direction_status": event.get("direction_status"),
                    "direction_confidence": event.get("direction_confidence"),
                    "direction_evidence": event.get("direction_evidence") or [],
                    "source_container": event.get("source_container"),
                    "target_container": event.get("target_container"),
                })
            if event.get("state_change_type") is not None:
                state_signals.append({
                    "event_id": event.get("event_id"),
                    "state_before": event.get("state_before"),
                    "state_after": event.get("state_after"),
                    "state_change_type": event.get("state_change_type"),
                    "state_confidence": event.get("state_confidence"),
                    "state_evidence": event.get("state_evidence") or [],
                })
            track_ids.extend(str(item) for item in (event.get("related_tracks") or event.get("involved_track_ids") or []))
            event_summaries.append(str(event.get("evidence_summary") or event.get("display_name") or event.get("event_type")))
        track_quality = self._track_quality_summary(track_ids, tracked_by_id)
        grade = self._bundle_grade(events, track_quality)
        return StepEvidenceBundle(
            protocol_step_id=node.protocol_step_id,
            linked_event_ids=[str(event.get("event_id")) for event in events if event.get("event_id")],
            linked_asset_paths=list(dict.fromkeys(linked_paths)),
            direction_signals=direction_signals,
            state_signals=state_signals,
            track_quality_summary=track_quality,
            evidence_grade=grade,
            evidence_summary=" | ".join(event_summaries[:4]) if event_summaries else "no_event_evidence",
        )

    @staticmethod
    def _track_quality_summary(track_ids: List[str], tracked_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        unique = list(dict.fromkeys(track_ids))
        tracks = [tracked_by_id[item] for item in unique if item in tracked_by_id]
        if not tracks:
            return {"track_count": 0, "avg_track_confidence": 0.0, "max_id_switch_risk": 1.0, "high_risk_track_ids": unique}
        avg = sum(float(track.get("track_confidence") or 0.0) for track in tracks) / len(tracks)
        max_risk = max(float(track.get("id_switch_risk") or 0.0) for track in tracks)
        high = [str(track.get("track_id")) for track in tracks if float(track.get("id_switch_risk") or 0.0) >= 0.45 or int(track.get("fragment_count") or 1) > 1]
        return {"track_count": len(tracks), "avg_track_confidence": round(avg, 4), "max_id_switch_risk": round(max_risk, 4), "high_risk_track_ids": high}

    @staticmethod
    def _bundle_grade(events: List[Dict[str, Any]], track_quality: Dict[str, Any]) -> str:
        if not events:
            return "weak"
        scores = [GRADE_SCORE.get(str(event.get("evidence_grade")), 0.35) for event in events]
        avg = sum(scores) / len(scores)
        if track_quality.get("avg_track_confidence", 0.0) < 0.45 or track_quality.get("max_id_switch_risk", 0.0) > 0.65:
            avg -= 0.18
        if avg >= 0.78:
            return "strong"
        if avg >= 0.48:
            return "medium"
        return "weak"

    def _decide(self, node: ProtocolStepNode, events: List[Dict[str, Any]], bundle: StepEvidenceBundle, previous_confirmed_index: int) -> StepPromotionDecision:
        issues: List[str] = []
        required = set(node.required_event_types)
        matched_types = {str(event.get("event_type")) for event in events}
        if required and not required.issubset(matched_types):
            issues.append("missing_required_event")
        if node.predecessor_ids and previous_confirmed_index < node.step_index - 1:
            issues.append("predecessor_not_confirmed")
        for field in node.critical_fields:
            if self._critical_missing(field, events):
                issues.append(f"missing_or_unstable:{field}")
        if bundle.track_quality_summary.get("avg_track_confidence", 0.0) < 0.45 or bundle.track_quality_summary.get("max_id_switch_risk", 0.0) > 0.7:
            issues.append("track_quality_too_low")
        score = self._candidate_score(events, bundle, issues)
        if issues:
            if "missing_required_event" in issues and not events:
                decision = "mark_inferred" if previous_confirmed_index >= node.step_index - 1 else "hold_for_review"
            else:
                decision = "hold_for_review"
        elif bundle.evidence_grade == "strong" and score >= 0.72:
            decision = "promote_confirmed"
        elif events:
            decision = "keep_candidate"
        else:
            decision = "mark_inferred"
        recommendation = {
            "promote_confirmed": "safe_to_promote_with_current_evidence",
            "keep_candidate": "retain_for_human_review_or_additional_evidence",
            "mark_inferred": "mark_as_inferred_do_not_confirm_without_event_evidence",
            "hold_for_review": "manual_review_required_before_step_promotion",
        }[decision]
        rationale = f"required={sorted(required)} matched={sorted(matched_types)} grade={bundle.evidence_grade} issues={issues}"
        return StepPromotionDecision(node.protocol_step_id, decision, round(score, 4), rationale, issues, recommendation)

    @staticmethod
    def _critical_missing(field: str, events: List[Dict[str, Any]]) -> bool:
        if not events:
            return True
        if field in {"source_container", "target_container"}:
            relevant = [event for event in events if event.get("event_type") == "liquid_transfer"]
            return not relevant or any(event.get(field) in (None, "", [], {}) for event in relevant)
        if field == "direction_status":
            relevant = [event for event in events if event.get("event_type") == "liquid_transfer"]
            return not relevant or any(event.get("direction_status") != "confirmed" for event in relevant)
        if field == "state_change_type":
            relevant = [event for event in events if event.get("event_type") == "container_state_change"]
            return not relevant or any(event.get("state_change_type") in (None, "", [], {}) for event in relevant)
        if field == "state_confidence":
            relevant = [event for event in events if event.get("event_type") == "container_state_change"]
            return not relevant or any(float(event.get("state_confidence") or 0.0) < 0.55 for event in relevant)
        if field == "actor_track_id":
            relevant = [
                event
                for event in events
                if event.get("event_type") in {"hand_object_interaction", "object_move", "liquid_transfer", "panel_operation"}
            ]
            return not relevant or any(event.get("actor_track_id") in (None, "", [], {}) for event in relevant)
        return any(event.get(field) in (None, "", [], {}) for event in events)

    @staticmethod
    def _candidate_score(events: List[Dict[str, Any]], bundle: StepEvidenceBundle, issues: List[str]) -> float:
        if not events:
            return 0.28 if "missing_required_event" in issues else 0.35
        event_score = sum(GRADE_SCORE.get(str(event.get("evidence_grade")), 0.35) - STATUS_PENALTY.get(str(event.get("review_status")), 0.15) for event in events) / len(events)
        track_score = float(bundle.track_quality_summary.get("avg_track_confidence") or 0.0)
        score = event_score * 0.65 + track_score * 0.25 + (0.1 if bundle.direction_signals or bundle.state_signals else 0.0)
        score -= min(0.3, 0.08 * len(issues))
        return max(0.0, min(1.0, score))

    def _candidate(self, experiment_id: str, node: ProtocolStepNode, events: List[Dict[str, Any]], decision: StepPromotionDecision, bundle: StepEvidenceBundle) -> StepCandidate:
        status_map = {
            "promote_confirmed": "confirmed",
            "keep_candidate": "candidate",
            "mark_inferred": "inferred",
            "hold_for_review": "needs_review",
        }
        raw = f"{experiment_id}:{node.protocol_step_id}:{','.join(bundle.linked_event_ids)}"
        return StepCandidate(
            step_candidate_id="stepcand_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16],
            experiment_id=experiment_id,
            protocol_step_id=node.protocol_step_id,
            protocol_step_name=node.protocol_step_name,
            matched_event_ids=bundle.linked_event_ids,
            matched_event_types=list(dict.fromkeys(str(event.get("event_type")) for event in events if event.get("event_type"))),
            candidate_score=decision.score,
            candidate_status=status_map[decision.decision],
            evidence_grade=bundle.evidence_grade,
            review_status="auto_confirmed" if decision.decision == "promote_confirmed" else ("low_confidence" if decision.decision == "hold_for_review" else "candidate_review"),
            reasoning_summary=decision.rationale,
        )

    @staticmethod
    def _out_of_order(nodes: List[ProtocolStepNode], candidates: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        event_by_id = {str(event.get("event_id")): event for event in events}
        observed: List[tuple[int, float, str]] = []
        node_index = {node.protocol_step_id: node.step_index for node in nodes}
        for cand in candidates:
            times = [float(event_by_id[event_id].get("start_time_sec") or 0.0) for event_id in cand.get("matched_event_ids", []) if event_id in event_by_id]
            if times:
                observed.append((node_index.get(cand["protocol_step_id"], 0), min(times), cand["protocol_step_id"]))
        out = []
        for prev, cur in zip(observed, observed[1:]):
            if cur[0] > prev[0] and cur[1] + 1.0 < prev[1]:
                out.append({"protocol_step_id": cur[2], "observed_time": cur[1], "previous_protocol_step_time": prev[1]})
        return out

    @staticmethod
    def _summary(experiment_id: str, nodes: List[ProtocolStepNode], candidates: List[Dict[str, Any]], decisions: List[Dict[str, Any]], missing, out_of_order, blocked) -> Dict[str, Any]:
        counts = {status: [c for c in candidates if c.get("candidate_status") == status] for status in ["confirmed", "candidate", "inferred", "needs_review"]}
        return {
            "schema_version": "step_bridge_summary.v1",
            "metadata_version": METADATA_VERSION,
            "experiment_id": experiment_id,
            "protocol_step_count": len(nodes),
            "confirmed_steps": counts["confirmed"],
            "candidate_steps": counts["candidate"],
            "inferred_steps": counts["inferred"],
            "needs_review_steps": counts["needs_review"],
            "missing_steps": missing,
            "out_of_order_steps": out_of_order,
            "blocked_steps": blocked,
            "promotion_decisions": decisions,
        }

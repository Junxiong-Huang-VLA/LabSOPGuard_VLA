from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import (
    METADATA_VERSION,
    OfficialStepLifecycleEvent,
    OfficialStepRecord,
    StepGovernanceDecision,
    StepReviewDecision,
    StepRevision,
    load_json,
    write_json,
)

ROLE_PERMISSIONS = {
    "reviewer": {"defer", "comment"},
    "approver": {"approve", "reject", "edit_and_approve", "defer", "comment"},
    "admin": {"approve", "reject", "edit_and_approve", "defer", "comment", "lock", "reopen", "supersede"},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str, *parts: Any) -> str:
    raw = ":".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


class StepReviewStore:
    def __init__(self, experiment_id: str, output_dir: str | Path) -> None:
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.official_path = self.output_dir / "official_steps.json"
        self.log_path = self.output_dir / "step_review_log.json"
        self.candidates_path = self.output_dir / "step_candidates.json"

    def ensure_outputs(self) -> None:
        if not self.official_path.exists():
            write_json(self.official_path, self._empty_official())
        if not self.log_path.exists():
            write_json(self.log_path, self._empty_log())

    def approve(
        self,
        *,
        step_candidate_id: str,
        decision: str,
        rationale: str,
        operator: str,
        edits: Optional[Dict[str, Any]] = None,
        operator_role: str = "approver",
    ) -> Dict[str, Any]:
        self._require_role(operator_role, decision)
        if decision not in {"approve", "edit_and_approve"}:
            raise ValueError("approve path only accepts approve or edit_and_approve")
        self.ensure_outputs()
        candidate, bundle = self._candidate_and_bundle(step_candidate_id)
        now = _now_iso()
        official = self._build_official(candidate, bundle, now, edits=edits)
        official_payload = load_json(self.official_path, self._empty_official())
        records = official_payload.setdefault("official_steps", [])
        existing_idx = next((idx for idx, item in enumerate(records) if item.get("official_step_id") == official["official_step_id"]), None)
        revision: Optional[Dict[str, Any]] = None
        if existing_idx is not None:
            previous = records[existing_idx]
            if previous.get("locked"):
                raise ValueError("official step is locked")
            official["version"] = int(previous.get("version") or 1) + 1
            records[existing_idx] = official
            revision = StepRevision(
                revision_id=_id("steprev", official["official_step_id"], official["version"], now),
                official_step_id=official["official_step_id"],
                previous_payload=previous,
                new_payload=official,
                change_reason=rationale or decision,
                operator=operator,
                operator_role=operator_role,
                created_at=now,
            ).to_dict()
        else:
            records.append(official)
        write_json(self.official_path, official_payload)
        review = self._append_review(step_candidate_id, decision, rationale, operator, operator_role, now, revision)
        return {"official_step": official, "review_decision": review, "revision": revision}

    def reject(self, *, step_candidate_id: str, rationale: str, operator: str, operator_role: str = "approver") -> Dict[str, Any]:
        self._require_role(operator_role, "reject")
        self.ensure_outputs()
        self._candidate_and_bundle(step_candidate_id)
        review = self._append_review(step_candidate_id, "reject", rationale, operator, operator_role, _now_iso(), None)
        return {"review_decision": review}

    def defer(self, *, step_candidate_id: str, rationale: str, operator: str, operator_role: str = "reviewer") -> Dict[str, Any]:
        self._require_role(operator_role, "defer")
        self.ensure_outputs()
        self._candidate_and_bundle(step_candidate_id)
        review = self._append_review(step_candidate_id, "defer", rationale, operator, operator_role, _now_iso(), None)
        return {"review_decision": review}

    def lock(self, *, official_step_id: str, operator: str, operator_role: str = "admin", rationale: str = "lock official step") -> Dict[str, Any]:
        self._require_role(operator_role, "lock")
        self.ensure_outputs()
        payload = load_json(self.official_path, self._empty_official())
        records = payload.setdefault("official_steps", [])
        idx = next((i for i, item in enumerate(records) if item.get("official_step_id") == official_step_id), None)
        if idx is None:
            raise ValueError("official step not found")
        previous = dict(records[idx])
        if previous.get("locked"):
            return {"official_step": previous, "revision": None}
        now = _now_iso()
        updated = dict(previous)
        updated["locked"] = True
        updated["lifecycle_status"] = "locked"
        updated["status"] = "locked"
        updated["version"] = int(updated.get("version") or 1) + 1
        records[idx] = updated
        revision = StepRevision(
            revision_id=_id("steprev", official_step_id, updated["version"], now),
            official_step_id=official_step_id,
            previous_payload=previous,
            new_payload=updated,
            change_reason=rationale,
            operator=operator,
            operator_role=operator_role,
            created_at=now,
        ).to_dict()
        write_json(self.official_path, payload)
        log = load_json(self.log_path, self._empty_log())
        log.setdefault("revisions", []).append(revision)
        lifecycle = self._lifecycle_event(official_step_id, str(previous.get("lifecycle_status") or previous.get("status") or "approved"), "locked", rationale, operator, operator_role, now)
        log.setdefault("lifecycle_events", []).append(lifecycle)
        log.setdefault("governance_decisions", []).append(self._governance_decision(official_step_id, "lock", rationale, operator, operator_role, now))
        write_json(self.log_path, log)
        return {"official_step": updated, "revision": revision, "lifecycle_event": lifecycle}

    def reopen(self, *, official_step_id: str, operator: str, operator_role: str = "admin", rationale: str = "reopen official step") -> Dict[str, Any]:
        self._require_role(operator_role, "reopen")
        return self._change_lifecycle(official_step_id, "reopened", operator, operator_role, rationale, locked=False)

    def supersede(
        self,
        *,
        official_step_id: str,
        operator: str,
        operator_role: str = "admin",
        rationale: str,
        replacement_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._require_role(operator_role, "supersede")
        self.ensure_outputs()
        payload = load_json(self.official_path, self._empty_official())
        records = payload.setdefault("official_steps", [])
        idx = next((i for i, item in enumerate(records) if item.get("official_step_id") == official_step_id), None)
        if idx is None:
            raise ValueError("official step not found")
        now = _now_iso()
        previous = dict(records[idx])
        superseded = dict(previous)
        superseded["lifecycle_status"] = "superseded"
        superseded["status"] = "superseded"
        superseded["locked"] = True
        superseded["version"] = int(superseded.get("version") or 1) + 1
        records[idx] = superseded
        replacement = dict(previous)
        replacement_payload = replacement_payload or {}
        replacement.update(replacement_payload)
        replacement["official_step_id"] = _id("officialstep", self.experiment_id, previous.get("protocol_step_id"), "supersede", now)
        replacement["status"] = str(replacement_payload.get("status") or "approved")
        replacement["lifecycle_status"] = str(replacement_payload.get("lifecycle_status") or "approved")
        replacement["locked"] = False
        replacement["version"] = 1
        replacement["supersedes_official_step_id"] = official_step_id
        replacement["created_at"] = now
        replacement["metadata_version"] = METADATA_VERSION
        records.append(replacement)
        revision = StepRevision(
            revision_id=_id("steprev", official_step_id, "supersede", now),
            official_step_id=official_step_id,
            previous_payload=previous,
            new_payload=superseded,
            change_reason=rationale,
            operator=operator,
            operator_role=operator_role,
            created_at=now,
        ).to_dict()
        lifecycle = self._lifecycle_event(official_step_id, str(previous.get("lifecycle_status") or previous.get("status") or "approved"), "superseded", rationale, operator, operator_role, now)
        write_json(self.official_path, payload)
        log = load_json(self.log_path, self._empty_log())
        log.setdefault("revisions", []).append(revision)
        log.setdefault("lifecycle_events", []).append(lifecycle)
        log.setdefault("governance_decisions", []).append(self._governance_decision(official_step_id, "supersede", rationale, operator, operator_role, now))
        write_json(self.log_path, log)
        return {"superseded_step": superseded, "replacement_step": replacement, "revision": revision, "lifecycle_event": lifecycle}

    def _candidate_and_bundle(self, step_candidate_id: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        payload = load_json(self.candidates_path, {})
        candidates = payload.get("step_candidates") or []
        candidate = next((item for item in candidates if item.get("step_candidate_id") == step_candidate_id), None)
        if not candidate:
            raise ValueError("step candidate not found")
        bundles = payload.get("evidence_bundles") or []
        bundle = next((item for item in bundles if item.get("protocol_step_id") == candidate.get("protocol_step_id")), {})
        return candidate, bundle

    def _build_official(self, candidate: Dict[str, Any], bundle: Dict[str, Any], created_at: str, edits: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        protocol_step_id = str(candidate.get("protocol_step_id"))
        official = OfficialStepRecord(
            official_step_id=_id("officialstep", self.experiment_id, protocol_step_id),
            experiment_id=self.experiment_id,
            protocol_step_id=protocol_step_id,
            protocol_step_name=str(candidate.get("protocol_step_name") or protocol_step_id),
            source_step_candidate_id=str(candidate.get("step_candidate_id")),
            status="approved",
            linked_event_ids=[str(item) for item in candidate.get("matched_event_ids") or []],
            evidence_bundle=bundle,
            created_at=created_at,
            locked=False,
            version=1,
            lifecycle_status="approved",
        ).to_dict()
        if edits:
            editable = {k: v for k, v in edits.items() if k not in {"official_step_id", "experiment_id", "version", "locked", "created_at"}}
            official.update(editable)
        return official

    def _append_review(
        self,
        step_candidate_id: str,
        decision: str,
        rationale: str,
        operator: str,
        operator_role: str,
        created_at: str,
        revision: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        log = load_json(self.log_path, self._empty_log())
        review = StepReviewDecision(
            review_decision_id=_id("stepreview", self.experiment_id, step_candidate_id, decision, created_at),
            step_candidate_id=step_candidate_id,
            decision=decision,
            rationale=rationale,
            operator=operator,
            operator_role=operator_role,
            created_at=created_at,
        ).to_dict()
        log.setdefault("review_decisions", []).append(review)
        if revision:
            log.setdefault("revisions", []).append(revision)
        write_json(self.log_path, log)
        return review

    def _empty_official(self) -> Dict[str, Any]:
        return {
            "schema_version": "official_steps.v1",
            "metadata_version": METADATA_VERSION,
            "experiment_id": self.experiment_id,
            "official_steps": [],
        }

    def _empty_log(self) -> Dict[str, Any]:
        return {
            "schema_version": "step_review_log.v1",
            "metadata_version": METADATA_VERSION,
            "experiment_id": self.experiment_id,
            "review_decisions": [],
            "revisions": [],
            "lifecycle_events": [],
            "governance_decisions": [],
        }

    @staticmethod
    def _require_role(operator_role: str, action: str) -> None:
        role = operator_role or "reviewer"
        if action not in ROLE_PERMISSIONS.get(role, set()):
            raise PermissionError(f"role {role} cannot perform {action}")

    def _change_lifecycle(self, official_step_id: str, to_status: str, operator: str, operator_role: str, rationale: str, *, locked: bool) -> Dict[str, Any]:
        self.ensure_outputs()
        payload = load_json(self.official_path, self._empty_official())
        records = payload.setdefault("official_steps", [])
        idx = next((i for i, item in enumerate(records) if item.get("official_step_id") == official_step_id), None)
        if idx is None:
            raise ValueError("official step not found")
        previous = dict(records[idx])
        now = _now_iso()
        updated = dict(previous)
        updated["lifecycle_status"] = to_status
        updated["status"] = to_status
        updated["locked"] = locked
        updated["version"] = int(updated.get("version") or 1) + 1
        records[idx] = updated
        revision = StepRevision(
            revision_id=_id("steprev", official_step_id, to_status, now),
            official_step_id=official_step_id,
            previous_payload=previous,
            new_payload=updated,
            change_reason=rationale,
            operator=operator,
            operator_role=operator_role,
            created_at=now,
        ).to_dict()
        lifecycle = self._lifecycle_event(official_step_id, str(previous.get("lifecycle_status") or previous.get("status") or "approved"), to_status, rationale, operator, operator_role, now)
        write_json(self.official_path, payload)
        log = load_json(self.log_path, self._empty_log())
        log.setdefault("revisions", []).append(revision)
        log.setdefault("lifecycle_events", []).append(lifecycle)
        log.setdefault("governance_decisions", []).append(self._governance_decision(official_step_id, to_status, rationale, operator, operator_role, now))
        write_json(self.log_path, log)
        return {"official_step": updated, "revision": revision, "lifecycle_event": lifecycle}

    @staticmethod
    def _lifecycle_event(official_step_id: str, from_status: str, to_status: str, rationale: str, operator: str, operator_role: str, created_at: str) -> Dict[str, Any]:
        return OfficialStepLifecycleEvent(
            lifecycle_event_id=_id("steplifecycle", official_step_id, from_status, to_status, created_at),
            official_step_id=official_step_id,
            from_status=from_status,
            to_status=to_status,
            rationale=rationale,
            operator=operator,
            operator_role=operator_role,
            created_at=created_at,
        ).to_dict()

    @staticmethod
    def _governance_decision(official_step_id: str, decision: str, rationale: str, operator: str, operator_role: str, created_at: str) -> Dict[str, Any]:
        return StepGovernanceDecision(
            governance_decision_id=_id("stepgov", official_step_id, decision, created_at),
            official_step_id=official_step_id,
            decision=decision,
            rationale=rationale,
            operator=operator,
            operator_role=operator_role,
            created_at=created_at,
        ).to_dict()

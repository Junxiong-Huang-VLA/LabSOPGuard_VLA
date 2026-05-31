"""Material approval gate and folder sync.

Ensures materials only enter the official library after human approval.
Approved materials are synced to a browsable folder structure.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

APPROVAL_SCHEMA_VERSION = "material_approval.v1"


class MaterialApprovalGate:
    """Controls the flow of materials from detection → review → approved library."""

    def __init__(self, experiment_dir: str | Path, library_root: Optional[str | Path] = None) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.experiment_id = self.experiment_dir.name
        self.review_queue_dir = self.experiment_dir / "_material_review_queue"
        self.library_root = Path(library_root) if library_root else self.experiment_dir.parent.parent / "material_references"
        self.approval_log_path = self.review_queue_dir / "approval_log.jsonl"

    def auto_publish_best_frames(self) -> Dict[str, Any]:
        """Auto-publish one best keyframe per event directly to library (no approval needed).

        This is the default: each physical interaction gets its peak representative
        frame in the official library. Clips and additional frames still require approval.
        """
        materials_dir = self.experiment_dir / "materials" / "events"
        if not materials_dir.exists():
            return {"published_count": 0}

        published = []
        for evt_dir in sorted(materials_dir.iterdir()):
            if not evt_dir.is_dir():
                continue
            event_json = evt_dir / "event.json"
            if not event_json.exists():
                continue

            data = json.loads(event_json.read_text(encoding="utf-8"))
            event = data.get("event", data)
            event_id = event.get("event_id", "")

            # Pick best frame: preview.jpg (= middle keyframe, highest score)
            best_frame = evt_dir / "preview.jpg"
            if not best_frame.exists():
                best_frame = evt_dir / "keyframe_02.jpg"
            if not best_frame.exists():
                candidates = sorted(evt_dir.glob("keyframe_*.jpg"))
                best_frame = candidates[0] if candidates else None

            if not best_frame or not best_frame.exists():
                continue

            # Sync best frame to library
            experiment_name = self._safe_name(self.experiment_id)
            event_type = event.get("event_type", "unknown")
            display_name = self._safe_name(event.get("display_name", event_id))

            target_dir = self.library_root / experiment_name / event_type
            target_dir.mkdir(parents=True, exist_ok=True)

            dst = target_dir / f"{display_name}_best.jpg"
            dst = self._unique_path(dst)
            shutil.copy2(best_frame, dst)

            published.append({
                "event_id": event_id,
                "display_name": event.get("display_name", ""),
                "best_frame_path": str(dst),
                "event_type": event_type,
            })

            # Log as auto-published
            self._log_approval(event_id, "auto_best_frame", "system", {"best_frame": str(dst)})

        logger.info("Auto-published %d best frames to library", len(published))
        return {"published_count": len(published), "items": published}

    def get_pending_materials(self) -> List[Dict[str, Any]]:
        """Get all materials pending review (clips + extra keyframes, not best frames)."""
        materials_dir = self.experiment_dir / "materials" / "events"
        if not materials_dir.exists():
            return []

        pending = []
        for evt_dir in sorted(materials_dir.iterdir()):
            if not evt_dir.is_dir():
                continue
            event_json = evt_dir / "event.json"
            if not event_json.exists():
                continue

            data = json.loads(event_json.read_text(encoding="utf-8"))
            event = data.get("event", data)

            # Check if already approved
            if self._is_approved(event.get("event_id", "")):
                continue

            clip = evt_dir / "clip.mp4"
            preview = evt_dir / "preview.jpg"
            keyframes = sorted(evt_dir.glob("keyframe_*.jpg"))

            pending.append({
                "event_id": event.get("event_id", ""),
                "event_type": event.get("event_type", ""),
                "display_name": event.get("display_name", ""),
                "start_time_sec": event.get("start_time_sec", 0),
                "end_time_sec": event.get("end_time_sec", 0),
                "confidence": event.get("confidence", 0),
                "evidence_grade": event.get("evidence_grade", ""),
                "review_status": "pending_review",
                "involved_objects": event.get("involved_objects", []),
                "has_clip": clip.exists(),
                "has_preview": preview.exists(),
                "keyframe_count": len(keyframes),
                "event_dir": str(evt_dir),
            })

        return pending

    def approve(self, event_id: str, reviewer: str = "operator") -> Dict[str, Any]:
        """Approve a material and sync to library folder."""
        evt_dir = self.experiment_dir / "materials" / "events" / event_id
        if not evt_dir.exists():
            return {"status": "error", "message": f"Event {event_id} not found"}

        event_json = evt_dir / "event.json"
        data = json.loads(event_json.read_text(encoding="utf-8"))
        event = data.get("event", data)

        # Sync to library folder
        synced_paths = self._sync_to_library(event, evt_dir)

        # Log approval
        self._log_approval(event_id, "approved", reviewer, synced_paths)

        logger.info("Material approved: %s → synced to library", event_id)
        return {
            "status": "approved",
            "event_id": event_id,
            "synced_paths": synced_paths,
        }

    def reject(self, event_id: str, reason: str = "", reviewer: str = "operator") -> Dict[str, Any]:
        """Reject a material (won't enter library)."""
        self._log_approval(event_id, "rejected", reviewer, [], reason=reason)
        logger.info("Material rejected: %s (reason: %s)", event_id, reason)
        return {"status": "rejected", "event_id": event_id, "reason": reason}

    def approve_all(self, reviewer: str = "operator") -> Dict[str, Any]:
        """Approve all pending materials."""
        pending = self.get_pending_materials()
        results = []
        for item in pending:
            result = self.approve(item["event_id"], reviewer=reviewer)
            results.append(result)
        return {
            "approved_count": sum(1 for r in results if r["status"] == "approved"),
            "results": results,
        }

    def get_approved_materials(self) -> List[Dict[str, Any]]:
        """Get list of already approved materials."""
        if not self.approval_log_path.exists():
            return []
        approved = []
        for line in self.approval_log_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("action") == "approved":
                approved.append(entry)
        return approved

    def _is_approved(self, event_id: str) -> bool:
        """Check if an event has already been approved."""
        if not self.approval_log_path.exists():
            return False
        for line in self.approval_log_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("event_id") == event_id and entry.get("action") == "approved":
                return True
        return False

    def _sync_to_library(self, event: Dict[str, Any], evt_dir: Path) -> Dict[str, str]:
        """Copy approved material assets to the organized library folder."""
        experiment_name = self._safe_name(self.experiment_id)
        event_type = event.get("event_type", "unknown")
        display_name = self._safe_name(event.get("display_name", event.get("event_id", "unnamed")))

        # Library structure: material_library/{experiment_name}/{event_type}/
        target_dir = self.library_root / experiment_name / event_type
        target_dir.mkdir(parents=True, exist_ok=True)

        synced = {}

        # Copy clip
        clip_src = evt_dir / "clip.mp4"
        if clip_src.exists():
            clip_dst = target_dir / f"{display_name}.mp4"
            clip_dst = self._unique_path(clip_dst)
            shutil.copy2(clip_src, clip_dst)
            synced["clip"] = str(clip_dst)

        # Copy keyframes
        for kf in sorted(evt_dir.glob("keyframe_*.jpg")):
            kf_dst = target_dir / f"{display_name}_{kf.stem}.jpg"
            kf_dst = self._unique_path(kf_dst)
            shutil.copy2(kf, kf_dst)
            synced[kf.stem] = str(kf_dst)

        # Copy preview
        preview_src = evt_dir / "preview.jpg"
        if preview_src.exists():
            preview_dst = target_dir / f"{display_name}_preview.jpg"
            preview_dst = self._unique_path(preview_dst)
            shutil.copy2(preview_src, preview_dst)
            synced["preview"] = str(preview_dst)

        return synced

    def _log_approval(
        self, event_id: str, action: str, reviewer: str,
        synced_paths: Any = None, reason: str = ""
    ) -> None:
        """Append to approval log."""
        self.review_queue_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "event_id": event_id,
            "action": action,
            "reviewer": reviewer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "synced_paths": synced_paths or {},
            "reason": reason,
            "schema_version": APPROVAL_SCHEMA_VERSION,
        }
        with self.approval_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_name(name: str) -> str:
        """Convert name to filesystem-safe string."""
        safe = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        safe = safe.replace("<", "").replace(">", "").replace('"', "")
        safe = safe.replace("|", "_").replace("?", "").replace("*", "")
        return safe[:80] if safe else "unnamed"

    @staticmethod
    def _unique_path(path: Path) -> Path:
        """If path exists, add suffix to make unique."""
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 2
        while True:
            new_path = parent / f"{stem}_{counter:02d}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

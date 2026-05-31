"""Tests for material approval gate."""
import json
from pathlib import Path

import pytest

from labsopguard.material_approval import MaterialApprovalGate


def _setup_experiment(tmp_path: Path) -> Path:
    """Create a minimal experiment directory with event materials."""
    exp_dir = tmp_path / "experiments" / "test-exp"
    materials = exp_dir / "materials" / "events"

    for evt_id in ["evt_001", "evt_002", "evt_003"]:
        evt_dir = materials / evt_id
        evt_dir.mkdir(parents=True)
        (evt_dir / "event.json").write_text(json.dumps({
            "event": {
                "event_id": evt_id,
                "event_type": "hand_object_interaction",
                "display_name": f"Test Event {evt_id}",
                "start_time_sec": 10.0,
                "end_time_sec": 15.0,
                "confidence": 0.8,
                "evidence_grade": "medium",
                "involved_objects": ["gloved_hand", "bottle"],
            }
        }), encoding="utf-8")
        (evt_dir / "clip.mp4").write_bytes(b"fake_video")
        (evt_dir / "preview.jpg").write_bytes(b"fake_image")
        (evt_dir / "keyframe_01.jpg").write_bytes(b"fake_kf1")
        (evt_dir / "keyframe_02.jpg").write_bytes(b"fake_kf2")

    return exp_dir


class TestMaterialApprovalGate:
    def test_get_pending_all(self, tmp_path: Path):
        exp_dir = _setup_experiment(tmp_path)
        gate = MaterialApprovalGate(exp_dir, library_root=tmp_path / "library")
        pending = gate.get_pending_materials()
        assert len(pending) == 3
        assert all(p["review_status"] == "pending_review" for p in pending)

    def test_approve_single(self, tmp_path: Path):
        exp_dir = _setup_experiment(tmp_path)
        library = tmp_path / "library"
        gate = MaterialApprovalGate(exp_dir, library_root=library)

        result = gate.approve("evt_001")
        assert result["status"] == "approved"
        assert "clip" in result["synced_paths"]

        # Check file was synced to library
        assert library.exists()
        clips = list(library.rglob("*.mp4"))
        assert len(clips) == 1

        # Check pending decreased
        pending = gate.get_pending_materials()
        assert len(pending) == 2

    def test_reject_single(self, tmp_path: Path):
        exp_dir = _setup_experiment(tmp_path)
        gate = MaterialApprovalGate(exp_dir, library_root=tmp_path / "library")

        result = gate.reject("evt_002", reason="Low quality")
        assert result["status"] == "rejected"

        # Library should be empty (nothing synced)
        library = tmp_path / "library"
        assert not library.exists() or not list(library.rglob("*.mp4"))

    def test_approve_all(self, tmp_path: Path):
        exp_dir = _setup_experiment(tmp_path)
        library = tmp_path / "library"
        gate = MaterialApprovalGate(exp_dir, library_root=library)

        result = gate.approve_all()
        assert result["approved_count"] == 3

        # All clips synced
        clips = list(library.rglob("*.mp4"))
        assert len(clips) == 3

        # No more pending
        pending = gate.get_pending_materials()
        assert len(pending) == 0

    def test_approved_not_pending(self, tmp_path: Path):
        exp_dir = _setup_experiment(tmp_path)
        gate = MaterialApprovalGate(exp_dir, library_root=tmp_path / "library")

        gate.approve("evt_001")
        pending = gate.get_pending_materials()
        assert not any(p["event_id"] == "evt_001" for p in pending)

    def test_library_folder_structure(self, tmp_path: Path):
        exp_dir = _setup_experiment(tmp_path)
        library = tmp_path / "library"
        gate = MaterialApprovalGate(exp_dir, library_root=library)

        gate.approve("evt_001")

        # Should have structure: library/{experiment_name}/{event_type}/
        exp_folder = library / "test-exp"
        assert exp_folder.exists()
        type_folder = exp_folder / "hand_object_interaction"
        assert type_folder.exists()
        assert any(f.suffix == ".mp4" for f in type_folder.iterdir())
        assert any(f.suffix == ".jpg" for f in type_folder.iterdir())

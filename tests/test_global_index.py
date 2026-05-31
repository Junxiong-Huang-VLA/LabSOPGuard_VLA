"""Tests for global cross-experiment material index."""
import json
import sqlite3
from pathlib import Path

import pytest

from labsopguard.global_index import GlobalMaterialIndex


def _create_experiment_index(exp_dir: Path, experiment_id: str, events: list) -> None:
    """Create a minimal material_index.sqlite for testing."""
    db_path = exp_dir / "material_index.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TABLE IF EXISTS event_materials")
    conn.execute("""
        CREATE TABLE event_materials (
            material_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            event_id TEXT,
            event_type TEXT,
            display_name TEXT,
            stable_name TEXT,
            actor_name TEXT,
            involved_objects_json TEXT,
            source_container_class TEXT,
            target_container_class TEXT,
            time_start REAL,
            time_end REAL,
            duration_sec REAL,
            evidence_grade TEXT,
            clip_path TEXT,
            preview_path TEXT,
            keyframe_count INTEGER,
            searchable_text TEXT
        )
    """)
    for evt in events:
        conn.execute(
            "INSERT INTO event_materials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                evt["material_id"], experiment_id, evt.get("event_id", ""),
                evt["event_type"], evt["display_name"], evt.get("stable_name", ""),
                evt.get("actor_name", "operator"),
                json.dumps(evt.get("objects", []), ensure_ascii=False),
                evt.get("source_container", ""), evt.get("target_container", ""),
                evt.get("time_start", 0.0), evt.get("time_end", 5.0),
                evt.get("duration", 5.0), evt.get("grade", "medium"),
                "", "", 3, evt.get("searchable", evt["display_name"]),
            ),
        )
    conn.commit()
    conn.close()


class TestGlobalMaterialIndex:
    def test_sync_experiment(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "experiments" / "exp-001"
        exp_dir.mkdir(parents=True)
        _create_experiment_index(exp_dir, "exp-001", [
            {"material_id": "mat_1", "event_type": "object_move", "display_name": "move bottle", "objects": ["bottle"]},
            {"material_id": "mat_2", "event_type": "hand_object_interaction", "display_name": "touch paper", "objects": ["paper"]},
        ])

        idx = GlobalMaterialIndex(tmp_path / "global.sqlite")
        count = idx.sync_experiment("exp-001", exp_dir)
        assert count == 2
        idx.close()

    def test_search_by_event_type(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "experiments" / "exp-001"
        exp_dir.mkdir(parents=True)
        _create_experiment_index(exp_dir, "exp-001", [
            {"material_id": "mat_1", "event_type": "object_move", "display_name": "move bottle"},
            {"material_id": "mat_2", "event_type": "liquid_transfer", "display_name": "pipette"},
        ])

        idx = GlobalMaterialIndex(tmp_path / "global.sqlite")
        idx.sync_experiment("exp-001", exp_dir)

        results = idx.search(event_type="liquid_transfer")
        assert len(results) == 1
        assert results[0]["display_name"] == "pipette"
        idx.close()

    def test_search_by_text(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "experiments" / "exp-001"
        exp_dir.mkdir(parents=True)
        _create_experiment_index(exp_dir, "exp-001", [
            {"material_id": "mat_1", "event_type": "object_move", "display_name": "move bottle", "searchable": "bottle glass reagent"},
            {"material_id": "mat_2", "event_type": "object_move", "display_name": "move paper", "searchable": "paper weighing"},
        ])

        idx = GlobalMaterialIndex(tmp_path / "global.sqlite")
        idx.sync_experiment("exp-001", exp_dir)

        results = idx.search(text="weighing")
        assert len(results) == 1
        assert "paper" in results[0]["display_name"]
        idx.close()

    def test_search_cross_experiment(self, tmp_path: Path) -> None:
        exp1 = tmp_path / "experiments" / "exp-001"
        exp1.mkdir(parents=True)
        exp2 = tmp_path / "experiments" / "exp-002"
        exp2.mkdir(parents=True)

        _create_experiment_index(exp1, "exp-001", [
            {"material_id": "mat_1", "event_type": "object_move", "display_name": "exp1 move"},
        ])
        _create_experiment_index(exp2, "exp-002", [
            {"material_id": "mat_2", "event_type": "object_move", "display_name": "exp2 move"},
        ])

        idx = GlobalMaterialIndex(tmp_path / "global.sqlite")
        idx.sync_experiment("exp-001", exp1)
        idx.sync_experiment("exp-002", exp2)

        results = idx.search(event_type="object_move")
        assert len(results) == 2
        experiment_ids = {r["experiment_id"] for r in results}
        assert experiment_ids == {"exp-001", "exp-002"}
        idx.close()

    def test_sync_all_experiments(self, tmp_path: Path) -> None:
        exps = tmp_path / "experiments"
        for name in ["exp-a", "exp-b", "exp-c"]:
            d = exps / name
            d.mkdir(parents=True)
            _create_experiment_index(d, name, [
                {"material_id": f"mat_{name}", "event_type": "object_move", "display_name": f"{name} event"},
            ])

        idx = GlobalMaterialIndex(tmp_path / "global.sqlite")
        results = idx.sync_all_experiments(exps)
        assert len(results) == 3
        assert idx.stats()["total_materials"] == 3
        idx.close()

    def test_stats(self, tmp_path: Path) -> None:
        idx = GlobalMaterialIndex(tmp_path / "global.sqlite")
        stats = idx.stats()
        assert stats["total_materials"] == 0
        assert stats["total_experiments"] == 0
        idx.close()

    def test_resync_replaces_old(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "experiments" / "exp-001"
        exp_dir.mkdir(parents=True)
        _create_experiment_index(exp_dir, "exp-001", [
            {"material_id": "mat_1", "event_type": "object_move", "display_name": "old event"},
        ])

        idx = GlobalMaterialIndex(tmp_path / "global.sqlite")
        idx.sync_experiment("exp-001", exp_dir)
        assert idx.stats()["total_materials"] == 1

        # Re-create with different data
        _create_experiment_index(exp_dir, "exp-001", [
            {"material_id": "mat_new1", "event_type": "liquid_transfer", "display_name": "new event 1"},
            {"material_id": "mat_new2", "event_type": "liquid_transfer", "display_name": "new event 2"},
        ])
        idx.sync_experiment("exp-001", exp_dir)
        assert idx.stats()["total_materials"] == 2
        idx.close()

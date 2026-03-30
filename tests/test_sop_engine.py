from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.monitoring.sop_engine import SOPComplianceEngine


def _detection(
    frame_id: int,
    ts: float,
    ppe: dict,
    pose_instances: int = 0,
    labels: list[str] | None = None,
):
    objs = [{"label": x} for x in (labels or [])]
    return SimpleNamespace(
        frame_id=frame_id,
        timestamp_sec=ts,
        ppe=ppe,
        objects=objs,
        actions=[],
        layer_outputs={"layer1_realtime_pose": {"pose_instances": pose_instances}},
    )


class SOPComplianceEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = {
            "ppe_requirements": {"must_wear": ["gloves", "goggles", "lab_coat"]},
            "violation_rules": {
                "missing_ppe": {"severity": "high", "message": "PPE is incomplete"}
            },
        }

    def test_missing_ppe_requires_human_presence(self) -> None:
        engine = SOPComplianceEngine(rules=self.rules, cooldown_seconds=10.0)
        det = _detection(
            frame_id=0,
            ts=0.0,
            ppe={"wear_gloves": False, "wear_goggles": False, "wear_lab_coat": False},
            pose_instances=0,
            labels=[],
        )
        violations = engine.update(det)
        self.assertEqual(len(violations), 0)

    def test_missing_ppe_checks_lab_coat_too(self) -> None:
        engine = SOPComplianceEngine(rules=self.rules, cooldown_seconds=10.0)
        det = _detection(
            frame_id=1,
            ts=1.0,
            ppe={"wear_gloves": True, "wear_goggles": True, "wear_lab_coat": False},
            pose_instances=1,
        )
        violations = engine.update(det)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].rule_id, "missing_ppe")
        self.assertEqual(violations[0].message, "PPE is incomplete")

    def test_cooldown_suppresses_repeated_rule(self) -> None:
        engine = SOPComplianceEngine(rules=self.rules, cooldown_seconds=10.0)
        d1 = _detection(
            frame_id=10,
            ts=2.0,
            ppe={"wear_gloves": False, "wear_goggles": False, "wear_lab_coat": False},
            pose_instances=1,
        )
        d2 = _detection(
            frame_id=11,
            ts=3.0,
            ppe={"wear_gloves": False, "wear_goggles": False, "wear_lab_coat": False},
            pose_instances=1,
        )
        self.assertEqual(len(engine.update(d1)), 1)
        self.assertEqual(len(engine.update(d2)), 0)

    def test_reset_clears_cooldown(self) -> None:
        engine = SOPComplianceEngine(rules=self.rules, cooldown_seconds=10.0)
        d = _detection(
            frame_id=20,
            ts=0.0,
            ppe={"wear_gloves": False, "wear_goggles": False, "wear_lab_coat": False},
            pose_instances=1,
        )
        self.assertEqual(len(engine.update(d)), 1)
        engine.reset()
        self.assertEqual(len(engine.update(d)), 1)


if __name__ == "__main__":
    unittest.main()

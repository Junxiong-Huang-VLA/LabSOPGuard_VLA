from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lab_vla.core.contracts import SkillStep, TaskCommand


class ContractTests(unittest.TestCase):
    def test_task_command_schema(self) -> None:
        cmd = TaskCommand(
            task_id="t1",
            instruction="pick and place",
            target_object="sample_container",
            source_zone="left",
            target_zone="right",
            constraints=["slow_motion"],
        )
        self.assertEqual(cmd.task_id, "t1")
        self.assertEqual(cmd.target_object, "sample_container")
        self.assertIn("slow_motion", cmd.constraints)

    def test_skill_step_schema(self) -> None:
        step = SkillStep(
            step_id="s1",
            skill_name="pick_object",
            command="pick",
            args={"target": "sample_container"},
            constraints=["obstacle_avoidance"],
        )
        self.assertEqual(step.command, "pick")
        self.assertEqual(step.args["target"], "sample_container")


if __name__ == "__main__":
    unittest.main()

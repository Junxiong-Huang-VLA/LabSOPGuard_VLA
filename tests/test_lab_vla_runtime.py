from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lab_vla.core.runtime import run_lab_vla


class RuntimeSmokeTests(unittest.TestCase):
    def test_runtime_closed_loop_mock(self) -> None:
        summary = run_lab_vla("configs/runtime/lab_vla_runtime.yaml")
        metrics = summary["metrics"]
        self.assertGreater(metrics["frames_total"], 0)
        self.assertGreater(metrics["plans_total"], 0)
        self.assertIn("last_result", summary)

        summary_path = PROJECT_ROOT / "outputs" / "runtime" / "lab_vla_summary.json"
        self.assertTrue(summary_path.exists())
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertIn("metrics", loaded)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

try:
    from integrated_system.app_integrated import create_app
except ModuleNotFoundError as exc:
    create_app = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(create_app is None, f"missing runtime dependency: {_IMPORT_ERROR}")
class IntegratedAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.client = self.app.test_client()

    def test_health(self) -> None:
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["status"], "ok")

    def test_analyze_missing_video(self) -> None:
        r = self.client.post("/api/analyze", data={})
        self.assertEqual(r.status_code, 400)

    def test_task_history_empty(self) -> None:
        r = self.client.get("/api/tasks?limit=5")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["ok"], True)
        self.assertIn("tasks", data)
        self.assertIn("total", data)
        self.assertIn("has_more", data)
        self.assertIn("status_counts", data)

    def test_task_history_filter(self) -> None:
        r = self.client.get("/api/tasks?limit=5&status=failed")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["ok"], True)
        self.assertIn("tasks", data)

    def test_retry_missing_task(self) -> None:
        r = self.client.post("/api/retry/not_exist")
        self.assertEqual(r.status_code, 404)
        data = r.get_json()
        self.assertEqual(data["ok"], False)


if __name__ == "__main__":
    unittest.main()

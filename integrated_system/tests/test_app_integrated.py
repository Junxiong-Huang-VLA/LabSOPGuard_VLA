from __future__ import annotations

import unittest

from integrated_system.app_integrated import create_app


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


if __name__ == "__main__":
    unittest.main()

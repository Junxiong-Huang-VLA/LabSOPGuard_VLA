from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient
from backend.main import app


def test_analysis_overview_contract_counts_are_same_source():
    client = TestClient(app)
    response = client.get("/api/v1/experiments/final_acceptance_e2e/analysis-overview")
    assert response.status_code == 200
    data = response.json()
    assert set(["experiment", "run", "readiness", "summary", "steps", "scene_summary", "alerts", "artifacts", "debug"]).issubset(data)
    assert data["run"]["run_id"]
    assert data["run"]["result_version"]
    assert data["summary"]["official_step_count"] == len(data["steps"]["official"])
    assert data["summary"]["candidate_step_count"] == len(data["steps"]["candidate"])
    assert data["summary"]["inferred_step_count"] == len(data["steps"]["inferred"])
    if data["run"]["status"] == "completed":
        assert all(data["readiness"].values())

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient
from backend.main import app
import backend.main as main


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


def test_latest_upload_e2e_benchmark_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    benchmark_root = tmp_path / "outputs" / "benchmarks"
    benchmark_root.mkdir(parents=True)
    (benchmark_root / "upload_e2e_latest.json").write_text(
        """
{
  "schema_version": "upload_e2e_benchmark.v2",
  "generated_at": "2026-05-20T00:00:00+00:00",
  "repeat_count": 2,
  "successful_ready_count": 2,
  "runs": [
    {
      "experiment_id": "run-1",
      "ready_for_demo": true,
      "client_end_to_end_sec": 40.0,
      "client_upload_http_sec": 5.0,
      "client_raw_upload_sec": 4.0,
      "client_analysis_wait_sec": 35.0,
      "client_sub_experiments_fetch_sec": 1.0,
      "client_materials_fetch_sec": 2.0,
      "segment_count": 3,
      "dual_view_segment_count": 3,
      "published_material_count": 12,
      "published_material_group_count": 3,
      "complete_dual_view_material_group_count": 3,
      "backend_timing": {
        "server_end_to_end_sec": 38.0,
        "algorithm_elapsed_sec": 30.0,
        "core_analysis_sec": 20.0
      }
    },
    {
      "experiment_id": "run-2",
      "ready_for_demo": true,
      "client_end_to_end_sec": 50.0,
      "client_upload_http_sec": 6.0,
      "client_raw_upload_sec": 5.0,
      "client_analysis_wait_sec": 44.0,
      "client_sub_experiments_fetch_sec": 1.5,
      "client_materials_fetch_sec": 2.5,
      "segment_count": 4,
      "dual_view_segment_count": 4,
      "published_material_count": 16,
      "published_material_group_count": 4,
      "complete_dual_view_material_group_count": 4,
      "backend_timing": {
        "server_end_to_end_sec": 48.0,
        "algorithm_elapsed_sec": 36.0,
        "core_analysis_sec": 22.0
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    client = TestClient(app)
    response = client.get("/api/v1/benchmarks/upload-e2e/latest", headers={"X-Operator-Role": "admin"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    assert payload["metrics"]["client_end_to_end_sec"]["sample_count"] == 2
    assert payload["metrics"]["client_end_to_end_sec"]["p50_sec"] == 45.0
    assert payload["metrics"]["client_end_to_end_sec"]["p90_sec"] == 49.0
    assert payload["metrics"]["client_raw_upload_sec"]["p50_sec"] == 4.5
    assert payload["metrics"]["client_sub_experiments_fetch_sec"]["p90_sec"] == 1.45
    assert payload["metrics"]["client_materials_fetch_sec"]["p50_sec"] == 2.25
    assert payload["target_status"]["target_p50_sec"] == 45.0
    assert payload["target_status"]["target_p90_sec"] == 60.0
    assert payload["target_status"]["passed"] is True
    assert payload["pipeline_contract"]["quality_gates"]["all_segments_have_dual_view"] is True
    assert payload["pipeline_contract"]["quality_gates"]["min_published_material_count"] == 12
    assert payload["pipeline_contract"]["quality_gates"]["min_complete_dual_view_material_group_count"] == 3
    assert payload["pipeline_contract"]["quality_gates"]["all_material_groups_have_dual_view_keyframe_and_clip"] is True

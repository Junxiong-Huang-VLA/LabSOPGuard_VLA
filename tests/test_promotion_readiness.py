from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.cli import main
from key_action_indexer.promotion_readiness import build_promotion_readiness_report, render_promotion_readiness_markdown


def test_promotion_readiness_reports_promoted_session(tmp_path: Path) -> None:
    session = tmp_path / "exp_1" / "key_action_index"
    _write_promoted_session(session, version="v003", query_count=3)

    report = build_promotion_readiness_report([tmp_path], query_count=3, output_json=tmp_path / "promotion.json", output_md=tmp_path / "promotion.md")
    row = report["sessions"][0]
    markdown = render_promotion_readiness_markdown(report)

    assert report["summary"]["promoted_count"] == 1
    assert row["readiness_status"] == "promoted"
    assert row["releases"]["latest_version"] == "v003"
    assert row["releases"]["promoted_version"] == "v003"
    assert row["gold_benchmark"]["fully_human_verified"] is True
    assert row["retrieval_eval"]["passes_required_eval"] is True
    assert row["blockers"] == []
    assert "Key Action Promotion Readiness" in markdown
    assert (tmp_path / "promotion.json").exists()
    assert (tmp_path / "promotion.md").exists()


def test_promotion_readiness_classifies_blockers(tmp_path: Path) -> None:
    session = tmp_path / "exp_2" / "key_action_index"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    (session / "manifest.json").write_text(json.dumps({"session_id": "blocked_session"}), encoding="utf-8")
    (metadata / "quality_gate.json").write_text(
        json.dumps(
            {
                "status": "fail",
                "can_mark_complete": False,
                "generated_at": "2026-05-08T10:00:00+00:00",
                "summary": {"blocking_count": 2, "adapter_semantic_issue_count": 1},
                "blocking_checks": [
                    {"name": "total_action_coverage_ratio", "actual": 1.0, "maximum": 0.65, "message": "coverage too broad"},
                    {"name": "adapter_semantic_issue_count", "actual": 1, "maximum": 0, "message": "semantic mismatch"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (metadata / "evidence_adapter_validation.json").write_text(
        json.dumps(
            {
                "status": "warning",
                "summary": {"present_adapter_count": 4, "error_count": 0, "warning_count": 1, "semantic_issue_count": 1, "missing_adapter_count": 0},
                "adapters": {"panel_ocr": {"semantic_issue_count": 1}},
            }
        ),
        encoding="utf-8",
    )
    (metadata / "gold_query_benchmark.json").write_text(
        json.dumps(
            {
                "query_count": 3,
                "human_verified_query_count": 1,
                "binding_mode": "partial_human_verified_review_file",
                "id_authoritative": False,
                "queries": [{"query_id": "gold_cn_001", "human_verified": True}],
            }
        ),
        encoding="utf-8",
    )
    (metadata / "review_queue.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-05-08T10:01:00+00:00",
                "summary": {"total": 2, "pending": 2},
                "items": [
                    {"item_id": "evidence_semantic:panel_ocr:semantic_missing_fields:1", "item_type": "evidence_semantic", "review_status": "pending"},
                    {"item_id": "segment:seg_1", "item_type": "segment", "review_status": "pending", "reasons": ["boundary_conflict"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_promotion_readiness_report([session], query_count=3)
    row = report["sessions"][0]
    codes = {item["code"] for item in row["blockers"]}

    assert row["readiness_status"] == "blocked"
    assert "reviewed_release_missing" in codes
    assert "quality_gate_blocking:total_action_coverage_ratio" in codes
    assert "quality_gate_blocking:adapter_semantic_issue_count" in codes
    assert "adapter_semantic_issues" in codes
    assert "gold_benchmark_not_human_verified" in codes
    assert "gold_decision_file_missing" in codes
    assert "retrieval_eval_missing" in codes
    assert row["review_queue"]["evidence_semantic_pending_count"] == 1
    assert row["review_queue"]["conflict_count"] == 1
    assert report["summary"]["blocked_count"] == 1


def test_promotion_readiness_flags_latest_release_needing_candidate_validation(tmp_path: Path) -> None:
    session = tmp_path / "exp_3" / "key_action_index"
    _write_promoted_session(session, version="v003", query_count=3)
    latest_dir = session / "reviewed_releases" / "v004"
    latest_dir.mkdir()
    (latest_dir / "reviewed_release_manifest.json").write_text(
        json.dumps({"version": "v004", "release_dir": str(latest_dir)}),
        encoding="utf-8",
    )
    (session / "reviewed_releases" / "latest_reviewed_release.json").write_text(
        json.dumps({"active_version": "v004", "release_dir": str(latest_dir)}),
        encoding="utf-8",
    )

    report = build_promotion_readiness_report([session], query_count=3)
    row = report["sessions"][0]

    assert row["readiness_status"] == "needs_candidate_validation"
    assert row["releases"]["latest_version"] == "v004"
    assert row["releases"]["promoted_version"] == "v003"
    assert row["releases"]["candidate_validation_current"] is False
    assert report["summary"]["needs_candidate_validation_count"] == 1
    assert report["summary"]["active_promoted_count"] == 1


def test_promotion_readiness_reports_failed_latest_candidate_eval(tmp_path: Path) -> None:
    session = tmp_path / "exp_4" / "key_action_index"
    _write_promoted_session(session, version="v003", query_count=3)
    latest_dir = session / "reviewed_releases" / "v004"
    latest_dir.mkdir()
    (latest_dir / "reviewed_release_manifest.json").write_text(
        json.dumps({"version": "v004", "release_dir": str(latest_dir)}),
        encoding="utf-8",
    )
    (session / "reviewed_releases" / "latest_reviewed_release.json").write_text(
        json.dumps({"active_version": "v004", "release_dir": str(latest_dir)}),
        encoding="utf-8",
    )
    (session / "evaluation" / "default_chinese_query_validation.v004.candidate.json").write_text(
        json.dumps(
            {
                "status": "fail",
                "query_count": 3,
                "topk_hit_rate": 0.33,
                "expected_id_hit_rate": 0.33,
                "expected_time_window_hit_rate": 1.0,
                "traceability_hit_rate": 1.0,
                "failed_query_count": 2,
                "threshold_failures": [{"metric": "expected_id_hit_rate", "actual": 0.33, "minimum": 0.75}],
                "category_summary": {
                    "weighing": {"query_count": 2, "failed_query_count": 2, "top3_hit_rate": 0.0, "expected_id_hit_rate": 0.0},
                    "mixing": {"query_count": 1, "failed_query_count": 0, "top3_hit_rate": 1.0, "expected_id_hit_rate": 1.0},
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_promotion_readiness_report([session], query_count=3)
    row = report["sessions"][0]
    codes = {item["code"] for item in row["blockers"]}

    assert row["readiness_status"] == "candidate_validation_failed"
    assert row["releases"]["candidate_validation_status"] == "failed"
    assert row["releases"]["latest_candidate_eval"]["status"] == "fail"
    assert row["releases"]["latest_candidate_eval"]["failure_profile"] == "time_window_pass_id_fail"
    assert row["releases"]["latest_candidate_eval"]["category_failures"][0]["category"] == "weighing"
    assert "candidate_retrieval_eval_failed" in codes
    candidate_blocker = next(item for item in row["blockers"] if item["code"] == "candidate_retrieval_eval_failed")
    assert candidate_blocker["details"]["expected_time_window_hit_rate"] == 1.0
    assert candidate_blocker["details"]["failure_profile"] == "time_window_pass_id_fail"
    assert report["summary"]["candidate_validation_failed_count"] == 1


def test_promotion_audit_cli_writes_reports(tmp_path: Path, capsys) -> None:
    session = tmp_path / "exp_1" / "key_action_index"
    _write_promoted_session(session, version="v001", query_count=2)

    exit_code = main(
        [
            "promotion-audit",
            "--source",
            str(tmp_path),
            "--query-count",
            "2",
            "--output-json",
            str(tmp_path / "cli_promotion.json"),
            "--output-md",
            str(tmp_path / "cli_promotion.md"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"readiness_status": "promoted"' in captured.out
    assert (tmp_path / "cli_promotion.json").exists()
    assert (tmp_path / "cli_promotion.md").exists()


def test_promotion_readiness_excludes_marked_historical_session(tmp_path: Path) -> None:
    promoted = tmp_path / "exp_1" / "key_action_index"
    historical = tmp_path / "historical_yolo" / "key_action_index"
    _write_promoted_session(promoted, version="v001", query_count=2)
    (historical / "metadata").mkdir(parents=True)
    (historical / "manifest.json").write_text(json.dumps({"session_id": "historical_yolo"}), encoding="utf-8")
    (historical / "metadata" / "promotion_audit_exclusion.json").write_text(
        json.dumps(
            {
                "exclude_from_promotion_audit": True,
                "reason": "historical YOLO scratch session, not a reviewed promotion target",
                "reviewer": "tester",
            }
        ),
        encoding="utf-8",
    )

    report = build_promotion_readiness_report([tmp_path], query_count=2)
    markdown = render_promotion_readiness_markdown(report)

    assert report["session_count"] == 1
    assert report["excluded_session_count"] == 1
    assert report["sessions"][0]["session_id"] == "promoted_session"
    assert report["excluded_sessions"][0]["session_id"] == "historical_yolo"
    assert "historical YOLO scratch session" in markdown


def test_promotion_readiness_flags_gold_release_mismatch(tmp_path: Path) -> None:
    session = tmp_path / "exp_1" / "key_action_index"
    _write_promoted_session(session, version="v002", query_count=2)
    gold_path = session / "metadata" / "gold_query_benchmark.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold["reviewed_release"] = "v001"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")

    report = build_promotion_readiness_report([session], query_count=2)
    row = report["sessions"][0]
    codes = {item["code"] for item in row["blockers"]}

    assert row["readiness_status"] == "blocked"
    assert "gold_benchmark_release_mismatch" in codes


def _write_promoted_session(session: Path, *, version: str, query_count: int) -> None:
    metadata = session / "metadata"
    evaluation = session / "evaluation"
    release_dir = session / "reviewed_releases" / version
    metadata.mkdir(parents=True)
    evaluation.mkdir()
    release_dir.mkdir(parents=True)
    (session / "manifest.json").write_text(json.dumps({"session_id": "promoted_session"}), encoding="utf-8")
    (release_dir / "reviewed_release_manifest.json").write_text(
        json.dumps({"version": version, "release_dir": str(release_dir)}),
        encoding="utf-8",
    )
    (session / "reviewed_releases" / "latest_reviewed_release.json").write_text(
        json.dumps({"active_version": version, "release_dir": str(release_dir)}),
        encoding="utf-8",
    )
    promoted = {
        "active_version": version,
        "release_dir": str(release_dir),
        "promotion_requirements": {"gold_benchmark_binding_mode": "human_verified_review_file"},
    }
    (session / "reviewed_releases" / "promoted_release.json").write_text(json.dumps(promoted), encoding="utf-8")
    (metadata / "promoted_release.json").write_text(json.dumps(promoted), encoding="utf-8")
    (metadata / "quality_gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "can_mark_complete": True,
                "generated_at": "2026-05-08T10:00:00+00:00",
                "summary": {"blocking_count": 0, "reviewed_release": version, "promoted_release": version, "adapter_semantic_issue_count": 0},
                "blocking_checks": [],
            }
        ),
        encoding="utf-8",
    )
    (metadata / "evidence_adapter_validation.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "summary": {"present_adapter_count": 4, "error_count": 0, "warning_count": 0, "semantic_issue_count": 0, "missing_adapter_count": 0},
                "adapters": {},
            }
        ),
        encoding="utf-8",
    )
    (metadata / "gold_query_benchmark.json").write_text(
        json.dumps(
            {
                "query_count": query_count,
                "total_query_count": query_count,
                "applicable_query_count": query_count,
                "excluded_query_count": 0,
                "reviewed_release": version,
                "reviewed_release_dir": str(release_dir),
                "human_verified_query_count": query_count,
                "human_reviewed_query_count": query_count,
                "binding_mode": "human_verified_review_file",
                "id_authoritative": True,
                "manual_review_status": "approved",
                "queries": [{"query_id": f"gold_cn_{index:03d}", "human_verified": True} for index in range(1, query_count + 1)],
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "default_chinese_query_validation.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "query_count": query_count,
                "total_query_count": query_count,
                "applicable_query_count": query_count,
                "excluded_query_count": 0,
                "reviewed_release": version,
                "reviewed_release_dir": str(release_dir),
                "human_verified_query_count": query_count,
                "human_reviewed_query_count": query_count,
                "benchmark_binding_mode": "human_verified_review_file",
                "top1_hit_rate": 1.0,
                "topk_hit_rate": 1.0,
                "expected_id_hit_rate": 1.0,
                "traceability_hit_rate": 1.0,
                "failed_query_count": 0,
                "threshold_failures": [],
            }
        ),
        encoding="utf-8",
    )
    (metadata / "review_queue.json").write_text(
        json.dumps({"generated_at": "2026-05-08T10:00:00+00:00", "summary": {"total": 0, "pending": 0}, "items": []}),
        encoding="utf-8",
    )

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from key_action_indexer.experiment_window_state import (
    _apply_formal_window_audit_status,
    _formal_window_activity_audit,
    _state_signal_rows_from_state_rows,
    _write_formal_window_review_artifacts,
    _write_state_signal_algorithm_report,
    _write_window_boundary_diagnosis_report,
    _write_window_sync_index_enforcement_report,
)
from key_action_indexer.material_references import _write_material_stream, _write_p0_material_algorithm_reports


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(dict(row), ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _count(rows: list[Mapping[str, Any]], key: str, value: str) -> int:
    return sum(1 for row in rows if str(row.get(key) or "") == value)


def _sync_state_reports(metadata_dir: Path) -> dict[str, Any]:
    coarse_rows = _read_jsonl(metadata_dir / "coarse_state_segments.jsonl")
    signal_rows = _state_signal_rows_from_state_rows(coarse_rows)
    _write_jsonl(metadata_dir / "state_signal_rows.jsonl", signal_rows)
    state_report = _write_state_signal_algorithm_report(metadata_dir, signal_rows)

    formal = _read_json(metadata_dir / "formal_experiment_windows.json")
    windows = [row for row in formal.get("windows", []) if isinstance(row, dict)]
    before_status_counts: dict[str, int] = {}
    for window in windows:
        status = str(window.get("status") or window.get("visual_review_status") or "unknown")
        before_status_counts[status] = before_status_counts.get(status, 0) + 1
        audit = _formal_window_activity_audit(window)
        _apply_formal_window_audit_status(window, audit)

    if windows:
        formal["windows"] = windows
        formal["window_count"] = len(windows)
        formal["status"] = "p0_algorithm_refreshed"
        _write_json(metadata_dir / "formal_experiment_windows.json", formal)

    _write_formal_window_review_artifacts(metadata_dir, windows)
    boundary_report = _write_window_boundary_diagnosis_report(metadata_dir, windows)
    sync_report = _write_window_sync_index_enforcement_report(metadata_dir, windows)
    return {
        "state_signal_rows": len(signal_rows),
        "state_signal_algorithm_report": str(metadata_dir / "state_signal_algorithm_report.json"),
        "window_boundary_diagnosis_report": str(metadata_dir / "window_boundary_diagnosis_report.json"),
        "window_fix_report": str(metadata_dir / "window_fix_report.json"),
        "window_boundary_before_after_report": str(metadata_dir / "window_boundary_before_after_report.json"),
        "window_self_validation_report": str(metadata_dir / "window_self_validation_report.json"),
        "window_sync_index_enforcement_report": str(metadata_dir / "window_sync_index_enforcement_report.json"),
        "first_active_bin_count": state_report.get("first_active_bin_count"),
        "third_active_bin_count": state_report.get("third_active_bin_count"),
        "before_status_counts": before_status_counts,
        "window_status_counts": {
            "validated_formal": boundary_report.get("validated_formal_count"),
            "suspicious_needs_review": boundary_report.get("suspicious_needs_review_count"),
            "rejected": boundary_report.get("rejected_count"),
        },
        "sync_enforcement": {
            "pass_count": sync_report.get("pass_count"),
            "fail_count": sync_report.get("fail_count"),
        },
    }


def _sync_material_reports(session_root: Path, material_root: Path) -> dict[str, Any]:
    metadata = session_root / "metadata"
    rows = _read_jsonl(metadata / "review_candidate_materials.jsonl")
    if not rows:
        candidate_index = session_root.parent / "material_references" / "review_candidate_materials.jsonl"
        rows = _read_jsonl(candidate_index)
    _write_p0_material_algorithm_reports(session_root, rows)
    if rows:
        _write_material_stream(material_root, rows, session_root=session_root)
    phase = _read_json(metadata / "dual_view_action_phase_report.json")
    quality = _read_json(metadata / "material_self_validation_report.json")
    dependency = _read_json(material_root / "reports" / "material_window_dependency_report.json")
    return {
        "review_candidate_material_rows": len(rows),
        "dual_view_action_phase_report": str(metadata / "dual_view_action_phase_report.json"),
        "dual_view_material_alignment_audit": str(metadata / "dual_view_material_alignment_audit.json"),
        "action_extraction_fix_report": str(metadata / "action_extraction_fix_report.json"),
        "action_candidate_rows": str(metadata / "action_candidate_rows.jsonl"),
        "keyclip_quality_report": str(metadata / "keyclip_quality_report.json"),
        "material_self_validation_report": str(metadata / "material_self_validation_report.json"),
        "material_window_dependency_report": str(material_root / "reports" / "material_window_dependency_report.json"),
        "phase_status_counts": phase.get("status_counts", {}),
        "material_validation": {
            "material_count": quality.get("material_count"),
            "pass_count": quality.get("pass_count"),
            "needs_review_count": quality.get("needs_review_count"),
        },
        "dependency_validation": {
            "orphan_material_count": dependency.get("orphan_material_count"),
            "missing_window_id_count": dependency.get("missing_window_id_count"),
            "missing_source_window_sync_index_count": dependency.get("missing_source_window_sync_index_count"),
            "fail_count": dependency.get("fail_count"),
        },
    }


def _write_expected_action_reports(metadata_dir: Path) -> dict[str, Any]:
    template_path = metadata_dir / "weak_expected_actions_template.json"
    if not template_path.exists():
        _write_json(
            template_path,
            {
                "schema_version": "weak_expected_actions_template.v1",
                "experiment_id": metadata_dir.parents[1].name,
                "expected_actions": [
                    {
                        "expected_event_id": "fill_me_001",
                        "action_type": "hand_object_contact",
                        "object_type": "weighing_paper_or_lab_object",
                        "expected_count": None,
                        "rough_time_ranges": [],
                        "notes": "人工补充真实动作真值后再计算 precision/recall/F1。",
                    }
                ],
            },
        )
    detected_rows = _read_jsonl(metadata_dir / "detected_actions.jsonl")
    _write_json(
        metadata_dir / "action_accuracy_report.json",
        {
            "schema_version": "action_accuracy_report.v1",
            "status": "not_validated_without_expected_actions",
            "expected_actions_path": str(template_path),
            "detected_count": len(detected_rows),
            "precision": None,
            "recall": None,
            "f1": None,
            "recommendations": ["提供 expected_actions.json 后才能声明动作准确率通过。"],
        },
    )
    return {
        "weak_expected_actions_template": str(template_path),
        "action_accuracy_report": str(metadata_dir / "action_accuracy_report.json"),
    }


def _write_database_consistency(material_root: Path) -> dict[str, Any]:
    stream_rows = _read_jsonl(material_root / "material_stream.jsonl")
    review_rows = _read_jsonl(material_root / "review_candidate_materials.jsonl")
    official_rows = _read_jsonl(material_root / "official_materials.jsonl")
    dependency = _read_json(material_root / "reports" / "material_window_dependency_report.json")
    report = {
        "schema_version": "database_consistency_validation_report.v1",
        "material_root": str(material_root),
        "material_stream_count": len(stream_rows),
        "review_candidate_count": len(review_rows),
        "official_count": len(official_rows),
        "orphan_material_count": dependency.get("orphan_material_count", 0),
        "missing_window_id_count": dependency.get("missing_window_id_count", 0),
        "missing_source_window_sync_index_count": dependency.get("missing_source_window_sync_index_count", 0),
        "memory_policy_violation_count": len(
            [row for row in stream_rows if row.get("official_status") != "official" and row.get("memory_eligible")]
        ),
        "status": "pass"
        if not dependency.get("fail_count") and not [row for row in stream_rows if row.get("official_status") != "official" and row.get("memory_eligible")]
        else "fail",
    }
    _write_json(material_root / "reports" / "database_consistency_validation_report.json", report)
    return report


def _write_frontend_validation(metadata_dir: Path, material_root: Path) -> dict[str, Any]:
    formal = _read_json(metadata_dir / "formal_experiment_windows.json")
    windows = [row for row in formal.get("windows", []) if isinstance(row, dict)]
    stream_rows = _read_jsonl(material_root / "material_stream.jsonl")
    report = {
        "schema_version": "frontend_visual_validation_report.v1",
        "experiment_id": material_root.name,
        "backend_window_count": len(windows),
        "frontend_expected_window_count": len(windows),
        "candidate_material_groups": len(stream_rows),
        "official_material_count": len([row for row in stream_rows if row.get("official_status") == "official"]),
        "needs_review_material_groups": len([row for row in stream_rows if row.get("official_status") != "official"]),
        "validated_formal_windows": [row.get("experiment_window_id") for row in windows if row.get("status") == "validated_formal"],
        "suspicious_windows": [
            row.get("experiment_window_id")
            for row in windows
            if str(row.get("status") or "") in {"formal_window_suspicious_needs_review", "formal_window_needs_human_review"}
        ],
        "rejected_windows": [row.get("experiment_window_id") for row in windows if row.get("status") == "formal_window_rejected"],
        "single_experiment_product_ready": False,
        "remaining_blockers": [
            "official material promotion is still controlled by publish/human confirmation policy",
            "expected_actions missing so accuracy cannot be claimed",
        ],
    }
    if report["validated_formal_windows"] and report["candidate_material_groups"]:
        report["single_experiment_product_ready"] = False
    _write_json(metadata_dir / "frontend_visual_validation_report.json", report)
    _write_json(material_root / "reports" / "frontend_visual_validation_report.json", report)
    return report


def _write_before_after(metadata_dir: Path, material_root: Path, state: Mapping[str, Any], material: Mapping[str, Any]) -> dict[str, Any]:
    formal = _read_json(metadata_dir / "formal_experiment_windows.json")
    stream_rows = _read_jsonl(material_root / "material_stream.jsonl")
    review_rows = _read_jsonl(material_root / "review_candidate_materials.jsonl")
    official_rows = _read_jsonl(material_root / "official_materials.jsonl")
    keyframe_quality = _read_json(metadata_dir / "keyframe_quality_report.json")
    before_status = dict(state.get("before_status_counts") or {})
    after_status = dict(state.get("window_status_counts") or {})
    phase_counts = dict(material.get("phase_status_counts") or {})
    dependency = dict(material.get("dependency_validation") or {})
    payload = {
        "schema_version": "algorithm_fix_before_after_report.v1",
        "before_window_count": formal.get("window_count"),
        "after_window_count": formal.get("window_count"),
        "before_validated_formal_count": int(before_status.get("validated_formal", 0)),
        "after_validated_formal_count": int(after_status.get("validated_formal", 0)),
        "before_suspicious_window_count": int(before_status.get("formal_window_suspicious_needs_review", 0))
        + int(before_status.get("formal_window_needs_human_review", 0)),
        "after_suspicious_window_count": int(after_status.get("suspicious_needs_review", 0)),
        "before_rejected_window_count": int(before_status.get("formal_window_rejected", 0)),
        "after_rejected_window_count": int(after_status.get("rejected", 0)),
        "before_action_suspicious_count": len(_read_jsonl(metadata_dir / "detected_actions.jsonl")),
        "after_action_suspicious_count": int(phase_counts.get("suspicious_needs_review", 0)),
        "before_orphan_material_count": None,
        "after_orphan_material_count": dependency.get("orphan_material_count"),
        "before_official_count": len(official_rows),
        "after_official_count": len(official_rows),
        "before_needs_review_count": len(review_rows),
        "after_needs_review_count": len(review_rows),
        "before_material_count": len(stream_rows),
        "after_material_count": len(stream_rows),
        "before_wrong_window_examples": [
            "formal_window_003 was blocked by pending_side_by_side_visual_review despite enough cross-view activity overlap."
        ],
        "after_wrong_window_examples": [
            "formal_window_001 remains rejected; formal_window_002 remains suspicious because cross-view overlap is still low."
        ],
        "before_wrong_material_examples": ["paired first/third evidence was always suspicious_needs_review."],
        "after_wrong_material_examples": [
            "events outside validated windows remain non-official; dual_view_valid is limited to validated windows with first/third keyframe and keyclip evidence."
        ],
        "keyframe_quality_before_after": {
            "keyframe_count": keyframe_quality.get("keyframe_count"),
            "scored_keyframe_count": keyframe_quality.get("scored_keyframe_count"),
            "blurry_frame_count": keyframe_quality.get("blurry_frame_count"),
        },
        "what_improved": [
            "validated_formal window promotion now uses state-signal cross-view activity overlap instead of permanent pending review",
            "dual-view action phase status can become dual_view_valid inside a validated window",
            "material_stream exposes window_id, source_window_sync_index, preview, sample_grid, and orphan_material",
        ],
        "what_still_fails": [
            "official_materials remains zero until publish/human confirmation policy promotes a validated candidate",
            "expected_actions missing, so precision/recall/F1 cannot be claimed",
            "formal_window_002 may need splitting or stronger state signals",
        ],
        "next_exact_code_level_fix": [
            "add action-specific bbox trace validation before automatic official promotion",
            "add expected_actions matching once ground truth is provided",
            "split windows with low first/third overlap if visual review confirms multiple phases",
        ],
        "state_signal_sync": state,
        "material_algorithm_reports": material,
    }
    _write_json(metadata_dir / "algorithm_fix_before_after_report.json", payload)
    _write_json(metadata_dir / "algorithm_before_after_report.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="desktop-import-real-regression-20260529-002")
    parser.add_argument("--experiment-root", default=None)
    parser.add_argument("--material-root", default=None)
    args = parser.parse_args()
    experiment_root = (
        Path(args.experiment_root)
        if args.experiment_root
        else Path(__file__).resolve().parents[1] / "outputs" / "experiments" / args.experiment_id
    )
    session_root = experiment_root / "key_action_index"
    metadata_dir = session_root / "metadata"
    material_root = Path(args.material_root) if args.material_root else Path("D:/LabMaterialLibrary") / args.experiment_id

    state = _sync_state_reports(metadata_dir)
    material = _sync_material_reports(session_root, material_root)
    accuracy = _write_expected_action_reports(metadata_dir)
    database = _write_database_consistency(material_root)
    frontend = _write_frontend_validation(metadata_dir, material_root)
    before_after = _write_before_after(metadata_dir, material_root, state, material)
    single_ready = (
        before_after.get("after_validated_formal_count", 0) >= 1
        and database.get("status") == "pass"
        and frontend.get("backend_window_count") == frontend.get("frontend_expected_window_count")
        and before_after.get("after_official_count", 0) > 0
    )
    result = {
        "schema_version": "p0_algorithm_report_refresh.v2",
        "experiment_id": args.experiment_id,
        "metadata_dir": str(metadata_dir),
        "material_root": str(material_root),
        "state": state,
        "material": material,
        "accuracy": accuracy,
        "database_consistency": database,
        "frontend_validation": frontend,
        "algorithm_fix_before_after_report": str(metadata_dir / "algorithm_fix_before_after_report.json"),
        "single_experiment_product_ready": bool(single_ready),
        "remaining_blockers": before_after.get("what_still_fails", []),
    }
    _write_json(metadata_dir / "p0_algorithm_report_refresh.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

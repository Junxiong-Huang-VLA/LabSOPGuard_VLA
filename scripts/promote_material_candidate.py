from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from key_action_indexer.experiment_action_ledger import apply_material_candidate_feedback


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sqlite_counts(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing"}
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("select official_status, count(*) from materials group by official_status").fetchall()
        official = conn.execute(
            "select material_id, official_status, memory_eligible from materials where official_status='official' order by material_id"
        ).fetchall()
        return {
            "status": "ok",
            "counts": {str(status): int(count) for status, count in rows},
            "official_materials": [
                {"material_id": str(material_id), "official_status": str(status), "memory_eligible": bool(memory)}
                for material_id, status, memory in official
            ],
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote or reject a LabMaterialLibrary needs_review material candidate.")
    parser.add_argument("--material-root", default=r"D:\LabMaterialLibrary")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--candidate-group-id")
    parser.add_argument("--material-id")
    parser.add_argument("--evidence-bundle-id")
    parser.add_argument("--candidate-id", action="append", dest="candidate_ids")
    parser.add_argument("--action", default="upgrade_to_official", choices=["confirm", "upgrade_to_official", "reject"])
    parser.add_argument("--reviewer", default="codex_p0_validation")
    parser.add_argument("--notes", default="")
    parser.add_argument("--reason-code")
    parser.add_argument("--reason")
    parser.add_argument("--refresh-corpus", action="store_true")
    args = parser.parse_args()

    material_root = Path(args.material_root)
    experiment_root = material_root / args.experiment_id
    before_ledger = _read_json(experiment_root / "experiment_action_ledger.json")
    before_stream = _read_jsonl(experiment_root / "material_stream.jsonl")
    before_official = _read_jsonl(experiment_root / "official_materials.jsonl")
    report = apply_material_candidate_feedback(
        material_root,
        args.experiment_id,
        candidate_group_id=args.candidate_group_id,
        material_id=args.material_id,
        evidence_bundle_id=args.evidence_bundle_id,
        candidate_ids=args.candidate_ids,
        action=args.action,
        reviewer=args.reviewer,
        notes=args.notes,
        reason_code=args.reason_code,
        reason=args.reason,
        refresh_corpus=args.refresh_corpus,
    )
    after_ledger = _read_json(experiment_root / "experiment_action_ledger.json")
    after_stream = _read_jsonl(experiment_root / "material_stream.jsonl")
    after_official = _read_jsonl(experiment_root / "official_materials.jsonl")
    sqlite_report = _sqlite_counts(experiment_root / "material_index.sqlite")

    validation = {
        "schema_version": "candidate_promotion_validation.v1",
        "experiment_id": args.experiment_id,
        "action": args.action,
        "candidate_group_id": args.candidate_group_id,
        "material_id": args.material_id,
        "evidence_bundle_id": args.evidence_bundle_id,
        "before": {
            "ledger_status": before_ledger.get("ledger_status"),
            "official_material_count": len(before_official),
            "stream_official_count": sum(1 for row in before_stream if row.get("official_status") == "official"),
        },
        "after": {
            "ledger_status": after_ledger.get("ledger_status"),
            "official_material_count": len(after_official),
            "stream_official_count": sum(1 for row in after_stream if row.get("official_status") == "official"),
            "confirmed_action_summary_count": len(after_ledger.get("confirmed_action_summary") or []),
        },
        "promotion_report": report,
        "sqlite": sqlite_report,
    }
    reports_root = experiment_root / "reports"
    _write_json(reports_root / "frontend_candidate_confirm_e2e_report.json", validation)
    _write_json(reports_root / "database_consistency_validation_report.json", {
        "schema_version": "database_consistency_validation_report.v1",
        "experiment_id": args.experiment_id,
        "status": "pass" if sqlite_report.get("status") == "ok" else "fail",
        "sqlite": sqlite_report,
        "material_stream_count": len(after_stream),
        "official_material_count": len(after_official),
        "official_materials_match_sqlite": sqlite_report.get("counts", {}).get("official", 0) == len(after_official),
    })
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

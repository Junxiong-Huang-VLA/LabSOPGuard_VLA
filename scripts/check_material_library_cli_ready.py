from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_STREAM_FIELDS = [
    "material_id",
    "evidence_bundle_id",
    "official_status",
    "action_type",
    "experiment_window_id",
    "global_timestamp_us",
    "source_window_sync_index",
    "first_keyframe",
    "third_keyframe",
    "first_keyclip",
    "third_keyclip",
    "keyframe_quality_score",
    "cli_ready_folder",
    "memory_eligible",
]


REQUIRED_FILE_FIELDS = [
    "source_window_sync_index",
    "first_keyframe",
    "third_keyframe",
    "first_keyclip",
    "third_keyclip",
    "cli_ready_folder",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_cli_ready_report(material_root: Path) -> dict[str, Any]:
    stream_path = material_root / "material_stream.jsonl"
    official_path = material_root / "official_materials.jsonl"
    review_path = material_root / "review_candidate_materials.jsonl"
    rows = _read_jsonl(stream_path)
    official_rows = _read_jsonl(official_path)
    review_rows = _read_jsonl(review_path)

    missing_fields: list[dict[str, Any]] = []
    missing_files: list[dict[str, Any]] = []
    memory_policy_violations: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        missing = [
            field
            for field in REQUIRED_STREAM_FIELDS
            if field not in row or row.get(field) in (None, "")
        ]
        if missing:
            missing_fields.append(
                {
                    "row": index,
                    "material_id": row.get("material_id"),
                    "missing": missing,
                }
            )
        for field in REQUIRED_FILE_FIELDS:
            value = row.get(field)
            if value and not Path(value).exists():
                missing_files.append(
                    {
                        "row": index,
                        "material_id": row.get("material_id"),
                        "field": field,
                        "path": value,
                    }
                )
        if row.get("official_status") == "needs_review" and row.get("memory_eligible") is True:
            memory_policy_violations.append(
                {
                    "row": index,
                    "material_id": row.get("material_id"),
                    "reason": "needs_review material must not be memory eligible",
                }
            )

    status = (
        "ready"
        if rows and not missing_fields and not missing_files and not memory_policy_violations
        else "needs_attention"
    )
    return {
        "schema_version": "cli_ready_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "material_root": str(material_root),
        "status": status,
        "material_stream": {
            "path": str(stream_path),
            "row_count": len(rows),
        },
        "official_materials": {
            "path": str(official_path),
            "row_count": len(official_rows),
        },
        "review_candidate_materials": {
            "path": str(review_path),
            "row_count": len(review_rows),
        },
        "checks": {
            "missing_required_fields": missing_fields,
            "missing_files": missing_files,
            "memory_policy_violations": memory_policy_violations,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a D:\\LabMaterialLibrary experiment folder for CLI/LLM consumption."
    )
    parser.add_argument("--material-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = build_cli_ready_report(args.material_root)
    output = args.output or args.material_root / "cli_ready_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.strict and report["status"] != "ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())

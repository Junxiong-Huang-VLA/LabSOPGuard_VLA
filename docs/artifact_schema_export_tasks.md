# Artifact Schema And Export Interface

## Scope

This entry covers the T-08/T-12 export interface work inside `src/key_action_indexer` only. It keeps the key action indexer independent from LabSOPGuard and does not add frontend, cloud camera, PTZ, or database infrastructure.

## Artifacts

The lightweight schema layer validates the current export-facing JSON/JSONL artifacts:

- `metadata/video_understanding.jsonl`
- `metadata/experiment_context.json`
- `metadata/experiment_process.json`
- `metadata/material_asset_catalog.jsonl`
- `metadata/human_confirmation_queue.jsonl`

The schemas are intentionally permissive about extra fields so existing payloads can evolve while stable IDs, timestamps, confidence, evidence refs, queue state, asset metadata, and process state remain checkable.

## CLI

Validate core artifacts:

```powershell
python -m key_action_indexer.cli validate-artifacts --session-dir <session_dir> --strict
```

Write a validation report:

```powershell
python -m key_action_indexer.cli validate-artifacts --session-dir <session_dir> --output <report.json>
```

Export a bundle for database/API/report ingestion:

```powershell
python -m key_action_indexer.cli export-artifacts --session-dir <session_dir> --output-dir <export_dir> --strict
```

The export command writes:

- `artifact_export_manifest.json`
- `artifact_export_summary.json`
- `artifact_export_hashes.json`
- `db_write_package.json`
- `retrieval_interface.json`
- `report_interface.json`
- `reusable_index/`
- `artifacts/<canonical artifact filename>`

The generated DB write package is a local handoff contract, not a database client. It preserves `version`, `source_session_id`, and `audit_trail` on session, step, timeline, evidence, asset, and confirmation rows. The retrieval interface points consumers at the reusable vector index and documents supported filters. The report interface exposes markdown and machine-readable report paths for downstream report systems.

## Current Limits

- Validation is a local JSON Schema subset implemented without `jsonschema` or other heavy dependencies.
- Missing artifacts are reported as validation errors in strict mode.
- The export interface copies and summarizes files; it does not write to SQL, LIMS, Vector DB, or remote APIs.
- Large JSONL artifacts remain as JSONL files in the bundle rather than being fully embedded in one monolithic payload.

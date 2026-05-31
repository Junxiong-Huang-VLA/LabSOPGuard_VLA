# evidence_package_manifest.json

**Version:** `evidence_package_manifest.v1`

Portable, read-only evidence package manifest for OpenClaw-style local consumers.
The package must be usable without the LabSOPGuard backend, database, service
ports, or original workstation paths.

## Required Top-Level Fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `schema_version` | string | yes | Must be `evidence_package_manifest.v1`. |
| `package_id` | string | yes | Stable identifier used in `package://{package_id}/...` URIs. |
| `experiment_id` | string | yes | Experiment/session identifier when available. |
| `created_at` | string | yes | ISO timestamp for package sidecar generation. |
| `path_mode` | string | yes | Must be `relative_to_package_root`. |
| `portable` | boolean | yes | Must be `true`. |
| `read_only_consumer` | boolean | yes | Must be `true`; consumers must not mutate LabSOPGuard state. |
| `backend_required` | boolean | yes | Must be `false`. |
| `entrypoints` | object | yes | Relative paths to package sidecar files. |
| `files` | array | no | Optional file integrity records for sidecars. |
| `provenance` | object | no | Relative or name-only source hints; must not be required to query. |

## Required Entrypoints

| Entrypoint | Target Schema | Notes |
| --- | --- | --- |
| `key_material_references_jsonl` | `key_material_reference_index.v1` rows | Queryable material references. |
| `physical_change_log_jsonl` | `physical_change.v1` rows | Hand-object, object movement, liquid transfer, panel/container events. |
| `time_alignment_json` | `time_alignment.v1` | Message/session/video time mapping. |

Recommended optional entrypoints:

| Entrypoint | Notes |
| --- | --- |
| `sqlite_index` | Local full-text SQLite index. |
| `material_index_jsonl` | Original material index, normalized by package builder. |
| `material_index_json` | Original JSON material index fallback. |

## Path Rules

- Entrypoint paths must be relative to the package root.
- Primary material references such as `stored_file`, `clip_path`,
  `formal_clip_path`, `preview_path`, and `keyframe_paths` must be relative
  package paths or `package://...` URIs.
- Absolute workstation paths are allowed only inside opaque historical payloads
  such as `payload_json`; they must not be the primary dereferenceable path.
- A copied package must validate and query from its new folder.

## Minimal Example

```json
{
  "schema_version": "evidence_package_manifest.v1",
  "package_id": "pkg_solid_001",
  "experiment_id": "exp_solid_001",
  "created_at": "2026-05-12T06:00:00+00:00",
  "path_mode": "relative_to_package_root",
  "portable": true,
  "read_only_consumer": true,
  "backend_required": false,
  "entrypoints": {
    "key_material_references_jsonl": "key_material_references.jsonl",
    "physical_change_log_jsonl": "physical_change_log.jsonl",
    "time_alignment_json": "time_alignment.json",
    "sqlite_index": "key_material_references.sqlite"
  }
}
```

## Validation

Use:

```powershell
python -m key_action_indexer.cli evidence-package-validate --package-root <package>
```

Strict mode also treats missing optional referenced keyframes/previews as errors:

```powershell
python -m key_action_indexer.cli evidence-package-validate --package-root <package> --strict
```

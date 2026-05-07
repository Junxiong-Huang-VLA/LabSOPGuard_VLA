# Semantic Material Publisher

Semantic Material Publisher is the derived asset publication layer for event-level materials.
It does not run detection, draw overlays, or modify the canonical event asset pack.

## Canonical Source Of Truth

The canonical source remains:

```text
outputs/experiments/{experiment_id}/materials/events/{event_id}/
  clip.mp4
  preview.jpg
  keyframe_01.jpg
  keyframe_02.jpg
  keyframe_03.jpg
  event.json
```

`event.json` and `material_index.sqlite` are the durable source of truth. Friendly
folders are derived views and must not be treated as the database.

## Published Archive

The publisher creates:

```text
outputs/experiments/{experiment_id}/published_materials/
  {operator_or_actor}/
    {event_type}/
      {timecode}_{stable_name}/
        clip.mp4
        preview.jpg
        keyframe_01.jpg
        keyframe_02.jpg
        keyframe_03.jpg
        event.json
        material_publish.json
```

Files are hard-linked where the platform allows it. If linking fails, the publisher
falls back to `copy2`. The operation is idempotent: rerunning publish updates metadata
and index rows without duplicating event records.

## Naming

Names are separated into three layers:

- `event_id`: unique event primary key from the event engine.
- `stable_name`: ASCII machine-stable name for paths and indexing.
- `display_name`: human-readable name, allowed to change.

Stable name format:

```text
{experiment_slug}__{event_type}__{source_to_target_or_object}__t{start}_{end}
```

Example:

```text
solid_weighing__liquid_transfer__bottle_to_beaker__t096_103
```

## Upload Manifest

The publisher writes:

```text
outputs/experiments/{experiment_id}/upload_manifest.json
```

This is the input contract for future OSS, S3, MinIO, NAS, or internal asset library
uploaders. `remote_url` is reserved and remains `null` until a real uploader is added.

## API

```text
POST /api/v1/experiments/{experiment_id}/materials/publish
GET  /api/v1/experiments/{experiment_id}/materials/published
GET  /api/v1/experiments/{experiment_id}/materials/upload-manifest
```

## Uploader Extension

Future uploaders should consume `upload_manifest.json` and write remote URLs back into
the manifest or a companion upload result file. They should not rewrite canonical event
assets or use friendly folders as the primary database.

Current uploader providers:

- `local`: copies clip and metadata into a local destination root.
- `nas`: same contract as local, intended for mounted NAS paths.
- `s3`, `minio`, `oss`: provider interfaces are present and return a not-configured result until credentials/client adapters are added.

API:

```text
POST /api/v1/experiments/{experiment_id}/materials/upload
```

Workspace published material aggregation:

```text
POST /api/v1/materials/published/reindex
GET  /api/v1/materials/published
```

The publisher also updates `official_steps.json` evidence bundles with
`published_material_refs` for linked event IDs. This is a derived reference update, not
a lifecycle change.

Display names can be enhanced from existing Qwen/VLM summary fields such as
`qwen_summary`, `semantic_summary`, `vlm_summary`, or `evidence_summary`. This only
changes `display_name`; `event_id` and `stable_name` remain stable.

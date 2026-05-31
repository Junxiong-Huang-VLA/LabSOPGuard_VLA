# 30-Day Video Memory T1-T14 Contracts

**Version:** `video_memory.t1_t14.contracts.v1`

This document defines the first testable data contracts for the 30-Day Video
Memory workstream. The implementation is expected to live behind
`key_action_indexer.video_memory` and to stay inside the existing physical
evidence pipeline: YOLO-backed candidates, micro-segments, VLM audit, durable
metadata, retrieval, and user feedback. It must remain runnable in dry-run mode
without real video files or ffmpeg.

## Public Python API

The contract tests in `tests/test_video_memory_contracts.py` expect these
symbols once the module is implemented:

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `EvidenceItem` | class/dataclass | Normalized memory unit for one observable physical event. |
| `validate_evidence_item` | function | Validate and normalize an Evidence Item into a JSON-compatible dict. |
| `build_vlm_cache_key` | function | Build a stable content key from model, prompt version, image refs, and evidence payload. |
| `VLMCache` | class | Local cache with `put(cache_key, response, metadata=...)` and `get(cache_key)` methods. |
| `score_evidence_clusters` | function | Score clusters from evidence items using recency, confidence, view support, and trace support. |
| `update_cluster_lifecycle` | function | Apply lifecycle policy and feedback events to scored clusters. |
| `build_partial_snapshot` | function | Emit a partial 30-day snapshot with truncation and skip metadata. |
| `answer_video_memory_query` | function | Answer from a snapshot and return claim-level evidence trace. |
| `run_feedback_update_job` | function | Plan/apply auditable feedback updates; dry-run must not mutate input snapshot. |

## Evidence Item

Schema version: `video_memory.evidence_item.v1`

Required top-level fields:

| Field | Type | Notes |
| --- | --- | --- |
| `evidence_id` | string | Stable item ID. |
| `cluster_id` | string | Cluster this item contributes to. |
| `session_id` | string | Source session ID. |
| `experiment_id` | string | Experiment ID when known. |
| `created_at` | ISO string | Creation time of the memory item. |
| `observed_at` | ISO string | Event observation time. |
| `time_range` | object | Includes `global_start_time`, `global_end_time`, `start_sec`, `end_sec`. |
| `action` | object | Includes `type`, `primary_object`, `secondary_objects`, `description`. |
| `physical_evidence` | object | Includes contact/change booleans, confidence, and uncertainty notes. |
| `views` | array | One row per view with clip/keyframe URIs and YOLO refs. |
| `vlm` | object | Includes `cache_key`, `model`, `prompt_version`, VLM summary, confidence. |
| `retrieval` | object | Includes `index_text`, `embedding_id`, score. |
| `trace` | object | Includes `decision_path`, `decision_trace`, source segment IDs. |
| `retention` | object | Includes `retention_days` and `expires_at`. |

Path rule: primary asset fields such as `clip_uri` and `keyframe_uri` must be
package-relative or `package://...` URIs. Absolute workstation paths must not be
used as dereferenceable memory assets.

## VLM Cache

Schema version: `video_memory.vlm_cache_entry.v1`

`build_vlm_cache_key` must be deterministic for identical semantic inputs,
regardless of keyword argument order or dict object identity. The key should be
derived from:

- model name
- prompt version
- image refs
- normalized evidence payload or evidence content hash

`VLMCache.get(cache_key)` returns:

```json
{
  "status": "hit",
  "cache_key": "stable-key",
  "response": {},
  "metadata": {
    "model": "qwen2.5-vl",
    "prompt_version": "physical-evidence-v1",
    "source_evidence_ids": ["ev_001"]
  }
}
```

Missing keys return `{"status": "miss", "cache_key": "..."}`.

## Cluster Lifecycle

Schema version: `video_memory.cluster.v1`

Required cluster fields:

| Field | Type | Notes |
| --- | --- | --- |
| `cluster_id` | string | Stable cluster ID. |
| `lifecycle_state` | string | One of `candidate`, `active`, `promoted`, `stale`, `archived`. |
| `score` | number | Final cluster score after scoring and feedback. |
| `score_reasons` | array | Auditable reasons such as `multi_view_support`, `accepted_feedback`. |
| `evidence_item_ids` | array | Evidence items supporting the cluster. |
| `last_observed_at` | ISO string | Latest observation time. |
| `archive_reason` | string | Required when `lifecycle_state` is `archived`. |

The default lifecycle policy uses a 30-day retention window. Clusters older than
that window are archived with `archive_reason:
older_than_retention_window`.

## Partial Snapshot

Schema version: `video_memory.partial_snapshot.v1`

Partial snapshots represent current usable memory under limits. They must not
claim completeness when retention, max item limits, missing cache entries, or
partial processing applies.

Required fields:

| Field | Type | Notes |
| --- | --- | --- |
| `snapshot_id` | string | Stable snapshot ID. |
| `snapshot_kind` | string | `partial` for incomplete snapshots. |
| `generated_at` | ISO string | Snapshot generation time. |
| `is_partial` | boolean | Must be `true` for partial snapshots. |
| `coverage` | object | Includes retention window and truncation reason. |
| `counts` | object | Includes source and included evidence item counts. |
| `evidence_items` | array | Included Evidence Items. |
| `clusters` | array | Included clusters. |
| `skipped_items` | array | Evidence IDs skipped due to retention, truncation, or invalidity. |

## Query Answer Evidence Trace

Schema version: `video_memory.query_answer.v1`

Every user-facing claim must map back to evidence. A cluster alone is not enough.
The trace must include:

- query text
- retrieved evidence rows
- final score breakdown
- evidence item IDs
- cluster IDs
- time ranges
- VLM cache keys
- limitations or uncertainty notes when support is incomplete

Claims with no evidence item IDs must be rejected or marked unsupported.

## Feedback Update Job

Schema version: `video_memory.feedback_update_job.v1`

Feedback events are append-only inputs. The job emits auditable operations such
as:

- `deprioritize_evidence_item`
- `rescore_cluster`
- `write_feedback_audit`
- optional `invalidate_vlm_cache`

Dry-run behavior is mandatory. When `dry_run=True`, the job returns
`job_status: planned` and must not mutate the supplied snapshot or source files.

## Scope Notes

- This contract does not introduce PTZ, five-camera orchestration, camera port
  mapping, MQTT, wireless SDKs, or `/api/v1/cameras`.
- YOLO bounding boxes are inputs to Evidence Items and clusters, not the final
  deliverable.
- All tests use metadata-only package URIs and fake payloads so they remain
  independent of real videos, GPU, ffmpeg, and backend services.

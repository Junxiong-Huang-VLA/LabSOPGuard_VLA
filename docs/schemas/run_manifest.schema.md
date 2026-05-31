# run_manifest.json Schema

**Version:** `run_manifest.v1`  
**Location:** `{output_dir}/run_manifest.json`

## Description

Pipeline execution manifest recording model versions, parameters, timing, and health for a single pipeline run. Written at the end of `run_pipeline()`.

## Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | yes | Always `"run_manifest.v1"` |
| `run_id` | string (UUID) | yes | Unique pipeline execution ID |
| `session_id` | string | yes | Session identifier from manifest |
| `dry_run` | boolean | yes | Whether this was a dry-run execution |
| `model_versions` | object | yes | Detector and model version info |
| `model_versions.detector_backend` | string | yes | `"motion"` or `"yolo_interaction"` |
| `model_versions.yolo_model_path` | string | no | Path to YOLO model file |
| `model_versions.yolo_conf` | float | yes | YOLO confidence threshold |
| `model_versions.yolo_iou` | float | yes | YOLO IoU threshold |
| `parameters` | object | yes | Detection parameters used |
| `parameters.sample_fps` | float | yes | Sampling frame rate |
| `parameters.start_threshold` | float | yes | Activity start threshold |
| `parameters.end_threshold` | float | yes | Activity end threshold |
| `parameters.merge_gap_sec` | float | yes | Gap merge threshold in seconds |
| `parameters.min_segment_duration_sec` | float | yes | Minimum segment length |
| `parameters.buffer_sec` | float | yes | Segment boundary buffer |
| `timing` | object | yes | Stage timing statistics |
| `timing.run_id` | string | yes | Same as top-level `run_id` |
| `timing.stage_count` | int | yes | Number of completed stages |
| `timing.total_duration_sec` | float | yes | Sum of all stage durations |
| `timing.stages` | array | yes | Per-stage timing entries |
| `timing.stages[].stage` | string | yes | Stage name |
| `timing.stages[].duration_sec` | float | yes | Stage wall-clock duration |
| `timing.stages[].inputs` | int | yes | Input count for stage |
| `timing.stages[].outputs` | int | yes | Output count for stage |
| `timing.stages[].errors` | int | yes | Error count (0 = success) |
| `alignment_health` | object | yes | Alignment health summary |
| `failure_nodes` | array | yes | Stages that completed with errors > 0 |

## Example

```json
{
  "schema_version": "run_manifest.v1",
  "run_id": "b6939af6-1fb0-41f7-b24b-b5c62081272d",
  "session_id": "exp_20260429",
  "dry_run": true,
  "model_versions": {
    "detector_backend": "yolo_interaction",
    "yolo_model_path": "models/best.pt",
    "yolo_conf": 0.25,
    "yolo_iou": 0.45
  },
  "parameters": {
    "sample_fps": 2.0,
    "start_threshold": 0.6,
    "end_threshold": 0.3,
    "merge_gap_sec": 5.0,
    "min_segment_duration_sec": 5.0,
    "buffer_sec": 2.0
  },
  "timing": {
    "run_id": "b6939af6-1fb0-41f7-b24b-b5c62081272d",
    "stage_count": 4,
    "total_duration_sec": 7.234,
    "stages": [
      {"stage": "validation", "duration_sec": 0.012, "inputs": 1, "outputs": 1, "errors": 0},
      {"stage": "detection", "duration_sec": 0.085, "inputs": 1, "outputs": 3, "errors": 0},
      {"stage": "micro_segmentation", "duration_sec": 0.026, "inputs": 3, "outputs": 3, "errors": 0},
      {"stage": "vector_index", "duration_sec": 0.057, "inputs": 6, "outputs": 6, "errors": 0}
    ]
  },
  "alignment_health": {"status": "healthy", "mean_offset_ms": 12.3, "jitter_ms": 3.1, "drift_events": 0},
  "failure_nodes": []
}
```

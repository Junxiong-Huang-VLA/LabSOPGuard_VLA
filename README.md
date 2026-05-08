# Key Action Indexer MVP

`key_action_indexer` is an offline Python MVP for turning long laboratory videos into time-anchored, searchable multimodal key action assets.

The main deliverable is a pipeline that reads long third-person and optional first-person experiment videos, aligns them with timestamped dialogue, detects continuous key action segments, extracts clips and keyframes, creates text descriptions, builds a local vector index, and returns clip paths from natural language queries.

## System Goal

The system solves this workflow:

1. Establish one `global_time` timeline for each experiment session.
2. Align third-person video, first-person video, and ASR transcript onto that timeline.
3. Detect when the operator is active at the lab bench.
4. Extract continuous key action clips, not only keyframes.
5. Attach dialogue, action metadata, clip paths, and timestamps to each segment.
6. Convert each segment into searchable text.
7. Build a local vector index for query-time retrieval.

This package deliberately does not implement a complex frontend, five-camera orchestration, port mapping, cloud model calls, PTZ, or wireless camera monitoring. PTZ tracking/control now lives outside this repository at `D:\PtzTracker`; multi-camera and wireless-video monitoring live at `D:\MultiCameraMonitor`.

## Inputs

- `SessionManifest` JSON file.
- Third-person long video.
- Optional first-person long video.
- Optional ASR/dialogue JSONL with `start_sec`, `end_sec`, and `text`.
- Optional workbench ROI.

Example manifest:

```json
{
  "session_id": "exp_20260429_172500",
  "session_start_time": "2026-04-29T17:25:00+08:00",
  "videos": {
    "third_person": {
      "path": "data/sessions/exp_20260429_172500/raw/third_person.mp4",
      "start_time": "2026-04-29T17:25:00+08:00",
      "fps": 30,
      "offset_sec": 0
    },
    "first_person": {
      "path": "data/sessions/exp_20260429_172500/raw/first_person.mp4",
      "start_time": "2026-04-29T17:25:02+08:00",
      "fps": 30,
      "offset_sec": 0
    }
  },
  "transcript": {
    "path": "examples/dialogue.example.jsonl",
    "start_time": "2026-04-29T17:25:00+08:00",
    "offset_sec": 0
  },
  "workbench_roi": {
    "x": 100,
    "y": 300,
    "w": 800,
    "h": 400
  },
  "output_dir": "data/sessions/exp_20260429_172500"
}
```

## Outputs

For each session, the pipeline writes:

```text
data/sessions/{session_id}/
  manifest.json
  transcript/aligned_transcript.jsonl
  cv_outputs/frame_scores.jsonl
  cv_outputs/detected_segments.jsonl
  clips/{segment_id}/third_person.mp4
  clips/{segment_id}/first_person.mp4
  keyframes/{segment_id}/*.jpg
  metadata/multimodal_alignment.jsonl
  metadata/key_action_segments.jsonl
  metadata/reviewed_dataset_manifest.json
  metadata/reviewed_segments.jsonl
  metadata/reviewed_micro_segments.jsonl
  metadata/reviewed_evidence.jsonl
  metadata/reviewed_export.json
  metadata/quality_gate.json
  metadata/evidence_adapter_validation.json
  metadata/vector_metadata.jsonl
  index/fallback_index.pkl
  reviewed_index/fallback_index.pkl
  index/faiss.index                # only when FAISS is available and dense embeddings are used
  index/docstore.jsonl
  index/index_config.json
```

## Directory Structure

```text
src/key_action_indexer/
  schemas.py
  config.py
  time_alignment.py
  transcript.py
  video_utils.py
  action_detector.py
  clip_extractor.py
  description_builder.py
  vector_index.py
  pipeline.py
  cli.py
tests/
examples/
data/sessions/
```

## Installation

Python 3.10+ is required.

```powershell
pip install -r requirements.txt
```

`requirements.txt` installs the local package in editable mode. `faiss-cpu` and `sentence-transformers` are optional. If FAISS is unavailable, the system saves and queries a local pickle fallback index. If sentence-transformers is not explicitly enabled, the MVP uses a local character n-gram TF-IDF backend when scikit-learn is available, otherwise a hashing fallback.

## Run Dry-Run MVP

Dry-run mode does not require real videos or ffmpeg. It creates mock detected segments, mock clips, mock keyframes, metadata, and a queryable vector index.

```powershell
python -m key_action_indexer.cli run --manifest examples/session_manifest.example.json --dry-run
```

## Query Example

```powershell
python -m key_action_indexer.cli query `
  --session-dir data/sessions/exp_20260429_172500 `
  --query "找一下使用天平称量的片段" `
  --top-k 3
```

Use `--index-dir data/sessions/exp_20260429_172500/index` when you only have the index folder. Query-validation configs are still supported explicitly:

```powershell
python -m key_action_indexer.cli query `
  --session-dir data/sessions/exp_20260429_172500 `
  --config examples/query_validation.example.json
```

Validation configs can include `thresholds` and a `quality_policy`; the output reports `status`, `acceptance_hit_rate`, failed queries, and threshold failures so it can be used as a regression gate.

Expected fields include:

- `segment_id`
- `score`
- `action_type`
- `global_start_time`
- `global_end_time`
- `third_person_clip`
- `first_person_clip`
- `evidence_level`
- `limitations`
- `index_text_preview`

## No-Label Run Health

Use the health command when you need a regression gate without GT annotations:

```powershell
python -m key_action_indexer.cli health `
  --session-dir data/sessions/exp_20260429_172500 `
  --query "balance weighing" `
  --output-json data/sessions/exp_20260429_172500/reports/run_health_report.json `
  --output-md data/sessions/exp_20260429_172500/reports/run_health_report.md
```

The report checks required artifacts, boundary confidence, total segment coverage, longest-segment ratio, vector index presence, clip/keyframe path integrity, context/input counts, queue size, and optional query smoke results. It does not require manual labels.

When a real run has no user text, AI replies, uploads, SOP records, or database records, seed a non-label operational context event from `manifest.json`, `video_info.json`, and `pipeline_summary.json` before running the health gate:

```powershell
python -m key_action_indexer.cli context-seed `
  --session-dir data/sessions/exp_20260429_172500
```

This writes `metadata/session_context_events.jsonl`. The event is marked `non_label_context: true`; it is not a manual label, SOP record, database record, or strong action confirmation.

After seeding or changing metadata on an existing run, refresh no-label derived artifacts so timeline, context, process reasoning, confirmation queue, QA, evaluation, and health reports agree:

```powershell
python -m key_action_indexer.cli refresh-derived `
  --session-dir data/sessions/exp_20260429_172500 `
  --query "balance weighing"
```

PowerShell helpers:

```powershell
.\scripts\check_key_action_outputs.ps1 -SessionDir data/sessions/exp_20260429_172500
.\scripts\run_key_action_smoke.ps1 -SessionDir data/sessions/exp_20260429_172500
.\scripts\frontend_smoke.ps1 -SessionDir data/sessions/exp_20260429_172500 -FrontendUrl http://127.0.0.1:5173
```

`run_key_action_smoke.ps1` runs Python compilation, `pytest -q`, key-action health, CLI query smoke, and frontend build/test. Add `-RunBrowserSmoke` when the local frontend and backend servers are already running.

## Run Real Video Pipeline

1. Place real videos under the session directory, for example:

```text
data/sessions/exp_20260429_172500/raw/third_person.mp4
data/sessions/exp_20260429_172500/raw/first_person.mp4
```

2. Update the manifest paths, start times, offsets, and ROI.

3. Run:

```powershell
python -m key_action_indexer.cli run --manifest examples/session_manifest.example.json
```

Real clip extraction requires `ffmpeg` on `PATH`. The extractor first tries stream copy and then falls back to H.264/AAC re-encoding.

## Detection Only

```powershell
python -m key_action_indexer.cli detect --manifest examples/session_manifest.example.json --dry-run
```

## Data Schema Summary

- `SessionManifest`: session ID, videos, transcript, ROI, output directory.
- `VideoSource`: name, path, start time, fps, offset.
- `TranscriptUtterance`: local transcript seconds and global timestamps.
- `DetectedSegment`: local third-person segment range, global time range, motion and active scores.
- `KeyActionSegment`: clip paths, local times per view, CV summary, dialogue context, text description, index info.
- `VectorMetadata`: embedding ID, segment ID, index text, timestamps, clip paths, dialogue, action type.

All pipeline outputs are JSON or JSONL serializable.

## MVP Limits

The current key action detector is a replaceable baseline:

- It analyzes ROI motion intensity in the third-person stream.
- `active_score` currently equals `motion_score`.
- It does not yet run hand detection, person detection, tool detection, or VLM reasoning.
- Real videos require OpenCV and ffmpeg.

The code is structured so `action_detector.py` can be replaced by a stronger detector that uses hand presence, person presence, tool recognition, and object interaction.

## Operational Notes

- YOLO-backed parent segments are refined with physical boundary support, so static lab context or isolated candidate transfer detections do not expand a key action to the whole video.
- Unsupported visual conclusions such as liquid stream, meniscus, OCR/readout, button/knob, and open/closed state remain candidate evidence until a dedicated model or OCR signal is present.
- Low-signal capability-gap candidates are kept in video understanding artifacts for retrieval and audit, but they are not all pushed into the human confirmation queue.
- Dry-run mode remains runnable without real videos or ffmpeg.

## Tests

```powershell
pytest
```

Covered behavior:

- local/global time conversion
- transcript alignment
- dialogue-to-segment matching
- segment merge/filter/buffer post-processing
- description generation and action-type inference
- vector build/query fallback
- full dry-run pipeline

## Extension Path

P1 replacements:

- Hand/person/tool detectors in `action_detector.py`.
- Manual anchor calibration for multimodal alignment.
- ASR-to-SOP step matching.
- Better action labels from local VLM or rule engines.

P2 replacements:

- Qdrant or Milvus for service-side indexing.
- Multi-session batch indexing.
- Minimal web query UI.

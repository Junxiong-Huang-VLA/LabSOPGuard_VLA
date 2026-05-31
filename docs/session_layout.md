# Session Layout

Each experiment session writes all outputs under one session root, normally `data/sessions/<session_id>` or the `output_dir` declared by the manifest.

## Required Directories

- `raw/`: raw input videos or references copied by upstream tools.
- `transcript/`: aligned ASR or transcript JSONL artifacts.
- `uploads/`: user supplied images, documents, screenshots, and parsed upload metadata.
- `cv_outputs/`: detector frame rows, frame scores, and detected segment rows.
- `clips/`: segment and micro-segment clips.
- `keyframes/`: segment and micro-segment keyframes.
- `metadata/`: JSON/JSONL contracts consumed by downstream understanding, reasoning, and retrieval.
- `index/`: vector index files and docstores.
- `reports/`: validation and smoke-test reports.
- `exports/`: final portable bundles and interface packages.

The implementation may also create `debug/` and `evaluation/` for local diagnostics and QA metrics.

## Core Files

- `manifest.json`: copied or generated canonical manifest.
- `video_info.json`: validation and source availability summary.
- `pipeline_summary.json`: end-to-end artifact manifest and counts.
- `metadata/video_sources.jsonl`: normalized video source rows for every view.
- `metadata/user_text_events.jsonl`: normalized user text events.
- `metadata/ai_reply_events.jsonl`: normalized AI reply events.
- `metadata/upload_events.jsonl`: normalized upload/media events.
- `transcript/aligned_transcript.jsonl`: ASR or transcript rows aligned to session time.
- `metadata/unified_multimodal_timeline.jsonl`: merged timeline for all modalities.

## Lifecycle

1. Initialize the directory tree before writing artifacts.
2. Copy or synthesize `manifest.json`.
3. Normalize input sources into `metadata/` and `transcript/`.
4. Generate detector outputs into `cv_outputs/`.
5. Produce clips, keyframes, metadata, indexes, reports, and exports.
6. Never delete unrelated artifacts during a module rerun.

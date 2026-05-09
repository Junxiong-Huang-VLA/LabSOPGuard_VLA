# Workspace consolidation

`LabSOPGuard` is the canonical project root going forward.

## Directory roles

- `LabSOPGuard`: main backend, frontend, experiment service, material stream, retrieval, synchronization, stream buffering, and semantic event detection.
- `lab_preprocessing`: legacy standalone preprocessing prototype. Its useful capabilities have been folded into `LabSOPGuard/src/labsopguard`:
  - stream/key-clip materialization: `stream_buffer.py` and `preprocessing.py`
  - timestamp alignment and drift correction: `time_sync.py`
  - event semantics: `semantic_events.py`
  - SQLite/FTS material querying: `retrieval.py`
- `.ultralytics`: runtime cache from YOLO/Ultralytics. New code points `YOLO_CONFIG_DIR` to `LabSOPGuard/.ultralytics`.
- `memory`: Codex/project-memory notes. A copy now lives under `LabSOPGuard/memory`; it is ignored by git because it is local operator context, not product code.

## Merge policy

Do not delete `D:\LabEmbodiedVLA\lab_preprocessing` until any external scripts or notebooks that import `src.pipelines.PreprocessingPipeline` have been migrated. New development should target `LabSOPGuard`.

Do not delete the old top-level `.ultralytics` or `memory` folders until after a clean run confirms no external tool still references them. They are safe to archive once `LabSOPGuard` is the only launch directory.

## New canonical runtime paths

- runtime scratch: `LabSOPGuard/.runtime`
- stream ring buffers: `LabSOPGuard/.runtime/stream_buffers`
- YOLO settings/cache: `LabSOPGuard/.ultralytics`
- project-memory copy: `LabSOPGuard/memory`

